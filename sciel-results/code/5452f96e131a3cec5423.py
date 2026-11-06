"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random

_START_TIME = None

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Safety guard: if we are approaching the 120s limit, fall back to a fast heuristic
    # to ensure we don't timeout. The deadline is 120s, we check against 115s.
    if _time.time() - _START_TIME > 115:
        # Fast fallback: pick the operation with the shortest duration on its preferred machine
        # to keep the system moving.
        min_dur = float('inf')
        best_idx = 0
        for i, op in enumerate(ready_operations):
            # Find min duration for this op
            min_op_dur = min(op["machine_options"].values())
            if min_op_dur < min_dur:
                min_dur = min_op_dur
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    if n_ops == 1:
        return 0

    # 1. Estimate machine load based on the current ready queue.
    # We assume that if we don't pick an op, it might be picked later or wait.
    # A simple proxy for "machine congestion" is the sum of durations of all ready ops
    # that can run on that machine.
    machine_load = {}
    for op in ready_operations:
        for m, d in op["machine_options"].items():
            if m not in machine_load:
                machine_load[m] = 0
            machine_load[m] += d

    # 2. Calculate a "Critical Ratio" for each operation.
    # For each operation, we consider its possible machines.
    # We want to pick the operation that is most "urgent" relative to the machine's load.
    # A low ratio implies the operation is small relative to the machine's total available work,
    # suggesting it can be processed quickly without adding much to the critical path,
    # or that the machine is not heavily loaded with other competing ready ops.
    
    # However, a better heuristic for Job Shop is often "Least Slack" or "Most Work Remaining" adjusted for machine.
    # Let's try a hybrid: 
    # Score = remaining_work / (1 + machine_load_of_best_machine)
    # This prioritizes jobs with more remaining work, but penalizes them if their best machine is already congested with other ready ops.
    
    best_score = -1
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        if rw == 0:
            # Should not happen for ready ops, but safety
            score = -1
        else:
            # Find the machine that minimizes the "penalty" or maximizes the score
            # Let's define the score for a specific (op, machine) pair.
            # If we pick this op on machine m, we contribute d to m's load.
            # The "efficiency" is how much work we clear per unit of machine time.
            # Actually, let's look at the "bottleneck" perspective.
            # If a machine has high load, we should avoid adding more work to it if possible,
            # UNLESS the job is large and needs to get moving.
            
            # Let's try: Score = remaining_work / (machine_load[m] + duration)
            # This rewards high remaining work and low machine congestion.
            
            max_op_score = -1
            for m, d in op["machine_options"].items():
                load = machine_load.get(m, 0)
                # Avoid division by zero, though load should be > 0 if d > 0
                denom = load + d
                if denom > 0:
                    s = rw / denom
                else:
                    s = 0
                if s > max_op_score:
                    max_op_score = s
            
            score = max_op_score
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
