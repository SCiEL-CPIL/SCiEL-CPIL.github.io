"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from collections import defaultdict

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Pre-compute for each operation: best machine, min duration, remaining work/ops
    best_machines = []
    min_durs = []
    rem_works = []
    rem_ops = []
    machine_options_list = []
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])
        machine_options_list.append(opts)

    # Deadline guard: fall back to LPT
    if not _elapsed_ok():
        idx = int(np.argmax(rem_works))
        return (idx, best_machines[idx])

    # Structural change: Lookahead Critical-Path Estimation
    # For each candidate operation, estimate the makespan impact by:
    # 1. Simulating its completion (subtract duration from remaining_work)
    # 2. Computing a congestion-weighted critical path estimate
    # Pick the operation whose completion reduces the estimated critical path most
    
    # Build machine -> list of (op_idx, duration) for all ready ops
    machine_ops = defaultdict(list)
    for i, opts in enumerate(machine_options_list):
        for m, d in opts.items():
            machine_ops[m].append((i, d))
    
    # For each operation, compute a "critical path score"
    # The idea: an operation is critical if it lies on a long chain of dependent work
    # We approximate by: for each op, compute sum of (remaining_work of its job) / (number of machines that job can use)
    # But we don't have job-level info directly, so use remaining_work as proxy
    
    # Alternative structural approach: Greedy makespan upper bound
    # For each candidate, estimate the makespan if we start it now on its best machine
    # Makespan >= max over machines of (current_time + sum of durations assigned to that machine)
    # We estimate by: for each machine, sum of min durations of all ready ops that can use it
    # Then pick the op that, when removed from consideration, most reduces this bound
    
    # Compute machine load estimates: for each machine, sum of min-duration of ops that can use it
    machine_load = {}
    all_machines = set()
    for opts in machine_options_list:
        for m in opts:
            all_machines.add(m)
    
    for m in all_machines:
        if m in machine_ops:
            # Sum of durations of all ready ops that can use this machine
            # Use the actual duration for each op on this machine
            load = sum(d for _, d in machine_ops[m])
            machine_load[m] = load
        else:
            machine_load[m] = 0
    
    # For each operation, estimate how much it contributes to the max machine load
    # If we start op i now, it occupies its best machine for min_durs[i] time
    # The makespan lower bound is max_m machine_load[m] + time (but ops overlap)
    # Better: estimate the "bottleneck contribution" of each op
    
    # Score each op by: 
    # - How much it reduces the estimated critical path
    # - Which machine it frees up (the one with highest load)
    
    scores = []
    for i in range(n):
        opts = machine_options_list[i]
        # Find the machine this op would use (best machine)
        bm = best_machines[i]
        dur = min_durs[i]
        
        # Estimate: if we start this op, the machine bm is busy for dur more time
        # The makespan is at least max over all machines of their total load
        # But since ops can run in parallel, the makespan is more like:
        # max_m (sum of durations of ops that MUST use m)
        
        # Key insight: an op is more "critical" if:
        # 1. It has few machine options (less flexibility)
        # 2. Its job has high remaining work (long tail)
        # 3. Its best machine is heavily loaded (bottleneck)
        
        num_options = len(opts)
        # Flexibility penalty: fewer options = more critical
        flexibility = num_options
        
        # Machine load on its best machine
        ml = machine_load.get(bm, 0)
        
        # Remaining work of the job (proxy for job length)
        rw = rem_works[i]
        
        # Composite: prioritize ops that are on critical paths
        # High remaining work + low flexibility + high machine load = critical
        score = rw * (1.0 / flexibility) * (1.0 + ml / (dur + 1e-9))
        
        scores.append((score, i))
    
    # Pick the highest-scoring operation (most critical)
    scores.sort(reverse=True)
    idx = scores[0][1]
    
    # Choose machine: use the machine that minimizes the resulting max load
    # For the chosen op, pick the machine that results in the lowest max machine load
    opts = machine_options_list[idx]
    best_m_final = min(opts, key=lambda m: (
        max(machine_load.get(m2, 0) + (opts[m] if m2 == m else 0) for m2 in all_machines),
        opts[m], m
    ))
    
    return (idx, best_m_final)
# EVOLVE-BLOCK-END
