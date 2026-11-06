"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math as _math

_START_TIME = None

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()

    # Deadline guard: evaluator kills at 150s; use 145s as safety margin.
    if _time.time() - _START_TIME > 145:
        # Fast fallback: shortest processing time
        best_idx = 0
        best_dur = float('inf')
        for i, op in enumerate(ready_operations):
            d = min(op["machine_options"].values())
            if d < best_dur:
                best_dur = d
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    if n_ops == 1:
        return 0

    # Precompute machine load from the ready queue
    machine_load = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            machine_load[m] = machine_load.get(m, 0) + d

    total_load = sum(machine_load.values()) + n_ops  # normalization factor

    best_score = -float('inf')
    best_idx = 0

    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        
        # Find minimum duration and corresponding machine
        min_dur = float('inf')
        best_machine = None
        for m, d in opts.items():
            if d < min_dur:
                min_dur = d
                best_machine = m
        
        # Critical path slack: remaining work divided by min duration
        rw = op["remaining_work"]
        slack = rw / (min_dur + 1e-9)
        
        # Load penalty: how loaded is the best machine relative to total load
        load_penalty = machine_load[best_machine] / total_load
        
        # Composite score: high slack, low load on best machine
        score = slack * (1.0 - load_penalty)

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
