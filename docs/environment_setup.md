# UCAgent 环境配置记录

> 本文档记录 2026-08-10 在本机 (WSL2 / Ubuntu 22.04) 完成 UCAgent 环境配置的完整过程,
> 包括安装步骤、踩坑记录与修复方法。适用于在 **conda Python 3.12 + conda-forge verilator 5.x** 环境下复现。

## 1. 环境概览

| 组件 | 版本 | 说明 |
|------|------|------|
| OS | Linux 6.18 (WSL2, Ubuntu 22.04) | |
| Python | 3.12.4 (miniconda base) | |
| verilator | 5.050 (conda-forge) | **需要 5.x 兼容修复,见 §3.2** |
| picker | 0.9.0-master (c100874) | 源码编译,安装到 `~/picker` |
| gcc / g++ | 11.4.0 | ≥11 满足 picker 要求 |
| swig | 4.4.1 | ≥4.2.0 满足要求 |
| cmake / make | 系统自带 | |
| Code Agent | qwen 0.21.8 (npm), claude, opencode | qwen 为推荐后端 |

## 2. 安装步骤

### 2.1 Python 依赖

```bash
pip3 install -r requirements.txt
```

- 安装内容含 langchain 1.3.14、langgraph、openai 2.53、pytoffee、toffee-test(gitlink.org.cn 源码)等。
- 与已装包有少量版本警告(protobuf/starlette/textual),不影响 UCAgent 运行。
- 验证: `python3 ucagent.py --help` 正常输出;`python3 ucagent.py --check` 全部 `[Found]`。

### 2.2 picker 编译安装

**注意: PyPI 上的 `picker` 包是无关的 Django 应用,不要 `pip install picker`。**

```bash
git clone https://gitlink.org.cn/XS-MLVP/picker.git --depth=1 /tmp/picker_x
cd /tmp/picker_x
make init        # 拉取 xcomm/slang/fmt 依赖(github 失败时 xcomm 自动回退 gitlink)
make
make install ARGS="-DCMAKE_INSTALL_PREFIX=$HOME/picker"
```

安装后:

```bash
export PATH="$HOME/picker/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/picker/lib:$LD_LIBRARY_PATH"
```

产物: `~/picker/bin/picker`、`~/picker/lib/libxspcomm.so`(同时解决了 xspcomm 依赖,它不在 PyPI 上)。

### 2.3 qwen-code 更新 (2026-08-10)

```bash
npm install -g @qwen-code/qwen-code@latest
```

- 版本: 0.11.0 → 0.21.8(npm 全局安装于 `~/.npm-global`)。
- 验证: `qwen -y -p "reply OK"`(即 UCAgent `setting.yaml` 中 qwen 后端的调用形式)返回 `OK`,与 UCAgent 兼容。
- 新版行为变化:
  - 首次运行初始化较慢(旧版秒回,新版可能 1 分钟内),属正常现象;
  - 无沙箱 yolo 模式会输出沙箱警告提示(`--yolo / approval-mode=yolo`),不影响执行。

## 3. 踩坑记录(重要)

### 3.1 `OPENAI_MODEL` 指向无权限模型(404)

- **现象**: `OPENAI_MODEL=deepseek-v3-2-251201` 对火山方舟 key 返回 `InvalidEndpointOrModel.NotFound`(qwen 和 curl 均 404)。
- **根因**: 该模型在账号下无访问权限(模型列表中存在但 key 不可用)。
- **修复**: 通过 `GET {OPENAI_BASE_URL}/models` 找到可用模型 `deepseek-v4-flash-ga-260731`,更新 `~/.bashrc`:

```bash
export OPENAI_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export OPENAI_API_BASE="$OPENAI_BASE_URL"   # UCAgent setting.yaml 读这个变量
export OPENAI_MODEL="deepseek-v4-flash-ga-260731"
```

- 验证: `qwen -y -p "reply OK"` 返回 `OK`。

### 3.2 conda verilator 的 `-V` 输出带 NUL 填充

- **现象**: picker/生成工程里 `verilator -V | grep ROOT | grep verilator | awk '{print $3}'` 探测 VERILATOR_ROOT 失败(grep 把输出当二进制,结果为空),导致 `find_package(verilator)` 找不到 config,构建失败。
- **根因**: conda-forge 的 verilator 二进制 `-V` 输出的 `VERILATOR_ROOT = ...` 行尾填充了大量 `\0` 字节。
- **修复**: 在 `~/picker/bin/`(PATH 最前)放置 wrapper,仅对 `-V` 过滤 NUL:

```bash
#!/bin/bash
# wrapper for conda-forge verilator: -V banner has NUL padding that breaks grep
if [ "$1" = "-V" ]; then
    /home/luyanfeng/miniconda3/bin/verilator -V | tr -d '\000'
else
    exec /home/luyanfeng/miniconda3/bin/verilator "$@"
fi
```

### 3.3 picker 模板与 verilator 5.x 不兼容(4 处)

以下修复同时应用到安装模板 `~/picker/share/picker/template/` 和源码模板 `/tmp/picker_x/template/`(升级 picker 后需重新应用)。

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | `python/CMakeLists.txt` | `include(dut.cmake)` 但 codegen 从不生成该文件 → `Unknown CMake command "XSPyTarget"` | 在 `python/` 模板目录新建 `dut.cmake`,内容 `include(cmake/verilator.cmake)` |
| 2 | `python/CMakeLists.txt` | SWIG include 只有 `./`,找不到父目录的 `dut_base.hpp` → `Unable to find 'dut_base.hpp'` | `include_directories(./)` → `include_directories(./ ../)` |
| 3 | `mem_direct/gen_addr.cpp` | `SET_SIZE(WData)` — verilator 5.x 移除了 `WData` typedef → `'WData' was not declared` | `SET_SIZE(WData)` → `SET_SIZE(EData)`(等价,同为 uint32_t) |
| 4 | `mem_direct/Makefile` | `LDLIBS` 缺 `-llz4`,verilator 5.x 的 FST 需要动态链接 lz4 → `undefined reference to LZ4_compressBound` | `LDLIBS += -lpthread -lz -llz4 ...` |

### 3.4 lz4 头文件与库缺失

- **现象**: 编译 verilator 的 FST 支持时报 `lz4.h: No such file or directory`,链接时报 `cannot find -llz4`。
- **修复**: 使用 conda 自带的 lz4(系统只有运行时库,无 dev 包)。构建时传参:

```bash
make CFLAGS="-I$HOME/miniconda3/include" LIBRARY_PATH="$HOME/miniconda3/lib"
```

**注意**: 
- 顶层 Makefile 有 `export CFLAGS :=`,会清空环境变量 CFLAGS,所以必须用**命令行传参**(命令行变量优先级高于 Makefile 内赋值)。
- **不要**用命令行传 `CXXFLAGS` / `LDFLAGS` — 生成工程的子 Makefile 用 `+=` 追加关键路径(`-I ${V_ROOT}/include`、`-I ../build/DPIAdder`、`-Wl,--whole-archive` 等),命令行变量会覆盖 `+=` 导致路径丢失。

### 3.5 其他环境坑

| 现象 | 根因 | 处理 |
|------|------|------|
| `make init_Adder` 最后一步报 `example.py` 不存在 | Makefile 遗留逻辑与新版 picker 目录结构不一致(python 工程在 `Adder/python/`,且 Adder 示例已无 `example.py`) | 不影响: picker export 已生成完整工程,直接编译 `Adder/` 顶层 Makefile 即可 |
| 非 TTY 下 `-hm` 模式立即退出 | `-hm`/`-tui` 会 `set_break(True)` → 进入 PDB 交互,stdout 重定向后 PDB EOF 退出 | 预期行为。MCP server 在 PDB interaction 中启动(`init_cmd`),`make mcp_Adder` 在真实终端使用即可 |
| 后台无 `-hm` 模式不启动 MCP server | `init_cmd`(含 `start_mcp_server`)只在 PDB interaction 中执行 | 预期行为: MCP 接口是给外部 Code Agent 用的 |

## 4. 验证结果

| 验证项 | 命令 | 结果 |
|--------|------|------|
| ucagent CLI | `python3 ucagent.py --check` | ✅ 全部 Found |
| qwen 模型调用 | `qwen -y -p "reply OK"` (0.21.8) | ✅ OK |
| picker export | `picker export Adder.v --rw 1 --sname Adder --tdir ... -c -w Adder.fst` | ✅ 工程生成 |
| verilator 构建 | `make CFLAGS=... LIBRARY_PATH=...` (mem_direct 模式) | ✅ libUTAdder.so + Adder_offset.yaml |
| SWIG 模块 | 同上 | ✅ `_UT_Adder.so` |
| DUT 运行 | `LD_LIBRARY_PATH=... python3 example.py` | ✅ exit 0 |
| MCP server | `-hm` 模式启动 | ✅ `FastMCP server started at 127.0.0.1:5000` |

## 5. 使用与测试

环境变量已固化到 `~/.ucagent_env`(README 推荐)和 `~/.bashrc`(PATH / LD_LIBRARY_PATH / OPENAI_*)。**新开终端生效**(当前已打开的终端需手动 `source ~/.ucagent_env`)。

> ⚠️ **注意**: README 的 `make mcp_Adder` 流程在此环境**不可直接用** — 其内部 `init_Adder` 步骤的 `example.py` 逻辑与新版 picker 工程结构不匹配会失败(见 §3.5)。workspace 生成后直接运行 ucagent 命令即可,无需走 make。

### 5.1 自动测试(qwen 后端,推荐)

```bash
source ~/.ucagent_env
cd ~/luyanfeng/fork-UCAgent

python3 ucagent.py output/workspace_Adder/ Adder -s -hm --tui \
  --mcp-server-no-file-tools --no-embed-tools --loop --backend=qwen
```

- UCAgent 自动拉起 qwen,qwen 通过 MCP 驱动完整验证流程(分析 RTL → 写测试 → 跑测试 → 出报告)。
- `--loop`: 完成后自动循环直到所有阶段完成;`ctrl+c` 中断进入交互,`status` 查看阶段进度。
- 完整验证约 6 个阶段,耗时可能几十分钟,会消耗 API 额度;快速验证可只跑前几个阶段。

### 5.2 手动测试(了解 MCP 协同机制)

终端 A(启动 UCAgent + MCP server):

```bash
source ~/.ucagent_env
cd ~/luyanfeng/fork-UCAgent
python3 ucagent.py output/workspace_Adder/ Adder -s -hm --tui \
  --mcp-server-no-file-tools --no-embed-tools
```

终端 B(启动 qwen 连接 MCP):

```bash
cd output/workspace_Adder
qwen
```

qwen 中输入任务提示词:

> 请通过工具`RoleInfo`获取你的角色信息和基本指导,然后完成任务。请使用工具`ReadTextFile`读取文件。你需要在当前工作目录进行文件操作,不要超出该目录。

### 5.3 测试结果

- 报告输出在 `output/workspace_Adder/` 下(`uc_test_report/`、`Adder_test_summary.md` 等)。
- TUI 快捷键: `ctrl+c` 中断命令进入交互,`q` 退出,`status` 查看状态,`help` 查看全部命令。

### 5.4 构建 DUT 时 lz4 报错的解决

```bash
cd output/workspace_Adder/Adder
make CFLAGS="-I$HOME/miniconda3/include" LIBRARY_PATH="$HOME/miniconda3/lib"
```

## 6. 遗留事项

- picker 的 `dut.cmake` 缺失、mem_direct 的 `WData`/`-llz4`、SWIG include 路径等问题应反馈给 XS-MLVP/picker 上游。
- `make init_Adder` 的 Makefile 流程与新版 picker/示例结构不匹配,若需使用建议同步修改 Makefile 或改用 Web Master Launch 流程。
