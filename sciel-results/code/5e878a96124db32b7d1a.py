"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

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
        # pick machine with smallest duration
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Pre-compute best machine and min duration for each operation
    # This is the "Model" input
    best_machines = []
    min_durs = []
    rem_works = []
    rem_ops = []
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        rem_works.append(op["remaining_work"])
        rem_ops.append(op["remaining_ops"])

    # Deadline guard: fall back to simple deterministic choice
    if not _elapsed_ok():
        idx = int(np.argmax(rem_works))
        return (idx, best_machines[idx])

    # Adaptive Heuristic with Load Balancing:
    # Score combines remaining work priority with machine load balancing
    # to reduce makespan bottleneck
    
    # Compute a composite score for each operation
    # Higher score = more urgent to schedule
    # Components:
    # 1. Remaining work (LPT): prioritize heavy jobs
    # 2. Current duration (SPT): prioritize quick completion to free machines
    # 3. Machine load balance: prefer machines that are less loaded
    
    # Normalize remaining work and durations for weighted combination
    max_rem_work = max(rem_works) if rem_works else 1
    max_min_dur = max(min_durs) if min_durs else 1
    
    scores = []
    for i in range(n):
        # LPT component: higher remaining work gets higher score
        lpt_score = rem_works[i] / max_rem_work if max_rem_work > 0 else 0
        
        # SPT component: shorter duration gets higher score
        spt_score = 1.0 - (min_durs[i] / max_min_dur) if max_min_dur > 0 else 0
        
        # Machine load balance: check if best machine is underutilized
        # We can't directly know machine loads, but we can use the fact that
        # choosing a machine with shorter duration for this op helps balance
        # If an op has a very short duration on its best machine, it's good for load balancing
        load_balance_score = 1.0 - (min_durs[i] / max_min_dur) if max_min_dur > 0 else 0
        
        # Weighted combination: emphasize LPT in early game, SPT in late game
        if n <= 3:
            # End game: prioritize quick completion
            w_lpt = 0.3
            w_spt = 0.5
            w_lb = 0.2
        else:
            # Early/mid game: prioritize heavy jobs
            w_lpt = 0.5
            w_spt = 0.3
            w_lb = 0.2
        
        score = w_lpt * lpt_score + w_spt * spt_score + w_lb * load_balance_score
        scores.append(score)
    
    # Select operation with highest composite score
    # Tie-break: higher remaining work, then shorter duration
    idx = max(
        range(n),
        key=lambda i: (
            scores[i],
            rem_works[i],
            -min_durs[i],
        ),
    )
    return (idx, best_machines[idx])
# EVOLVE-BLOCK-END
