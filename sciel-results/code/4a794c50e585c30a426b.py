"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START = _time.time()
_DEADLINE = 145.0  # safety margin under 150s wall-clock


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def _min_machine(opts):
    """Return (machine, duration) for the machine with minimum duration, tie-break by machine id."""
    best_m = None
    best_d = float('inf')
    for m, d in opts.items():
        if d < best_d or (d == best_d and best_m is not None and m < best_m):
            best_d = d
            best_m = m
    return best_m, best_d


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        best_m, _ = _min_machine(opts)
        return (0, best_m)

    # Precompute for each operation:
    # - best machine (min duration)
    # - min duration
    # - max duration
    # - spread
    # - remaining work
    # - remaining ops
    # - number of machine options (constraint level)
    
    best_machines = []
    min_durs = []
    max_durs = []
    spreads = []
    rem_works = []
    rem_ops = []
    opt_counts = []
    
    for op in ready_operations:
        opts = op["machine_options"]
        bm, bd = _min_machine(opts)
        # Calculate max duration for spread
        wd = max(opts.values())
        best_machines.append(bm)
        min_durs.append(bd)
        max_durs.append(wd)
        spreads.append(wd - bd)
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])
        opt_counts.append(len(opts))

    # Deadline guard
    if not _elapsed_ok():
        # Fallback: pick the one with most remaining work
        idx = int(np.argmax(rem_works))
        return (idx, best_machines[idx])

    # Composite scoring:
    # 1. Most Constrained First: Fewer machine options (lower count) is prioritized.
    #    This reduces the risk of delaying operations that have limited alternatives.
    # 2. Max Remaining Work (MWR): Prioritize jobs with more total work remaining.
    # 3. Shortest Processing Time (SPT): Prioritize operations with shorter min duration
    #    to keep machines cycling faster.
    # 4. Fewer Remaining Ops: Finish jobs sooner to free up job resources.
    
    order = sorted(
        range(n),
        key=lambda i: (
            opt_counts[i],       # Ascending: 1 option is most constrained
            -rem_works[i],       # Descending: More work first
            min_durs[i],         # Ascending: Shorter task first
            rem_ops[i],          # Ascending: Closer to completion first
        ),
    )

    idx = order[0]
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
