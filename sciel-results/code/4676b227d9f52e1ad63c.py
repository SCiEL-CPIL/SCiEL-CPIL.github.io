"""Job-shop dispatch rule: Balanced LRPT with Load-Aware Penalty."""
# EVOLVE-BLOCK-START
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 140.0  # safety margin well before the 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    1. Prioritize jobs with high Remaining Work (LRPT) to clear critical paths.
    2. Penalize operations that target heavily contended machines to balance load.
    3. Slight preference for shorter durations (SPT) to reduce queue length on the chosen machine.
    4. Small bonus for jobs with fewer remaining operations (urgency of completion).
    """
    if not ready_operations:
        return 0
    if _deadline_exceeded():
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Step 1: Determine the "best" machine for each operation (min duration)
    # and calculate machine loads based on these best assignments.
    best_machines = [None] * n
    best_durations = [float('inf')] * n
    machine_loads = {}

    for i in range(n):
        op = ready_operations[i]
        mo = op.get("machine_options", {})
        if not mo:
            continue
        
        min_dur = float('inf')
        min_m = None
        for m, d in mo.items():
            if d < min_dur:
                min_dur = d
                min_m = m
        
        best_machines[i] = min_m
        best_durations[i] = min_dur
        
        if min_m is not None:
            machine_loads[min_m] = machine_loads.get(min_m, 0) + 1

    max_load = max(machine_loads.values()) if machine_loads else 1
    if max_load == 0:
        max_load = 1

    # Weights
    W_RW = 1.0       # Weight for Remaining Work (LRPT)
    W_LOAD = 5.0     # Weight for Machine Load Penalty
    W_DUR = 0.1      # Weight for Duration (SPT)
    W_URGENCY = 1.0  # Weight for Urgency (1/remaining_ops)

    best_idx = 0
    best_score = float("-inf")

    for i in range(n):
        op = ready_operations[i]
        rw = float(op.get("remaining_work", 0) or 0)
        rem_ops = int(op.get("remaining_ops", 1) or 1)
        
        if rem_ops < 1:
            rem_ops = 1

        m = best_machines[i]
        d = best_durations[i]
        
        if m is None:
            # Fallback if no machine options (shouldn't happen)
            score = rw
        else:
            load = machine_loads.get(m, 0)
            norm_load = load / max_load
            
            # Urgency term: smaller remaining ops count is better (higher score)
            urgency = W_URGENCY / rem_ops
            
            # Load penalty: higher load is worse (lower score)
            load_penalty = W_LOAD * norm_load
            
            # Duration penalty: longer duration is worse (lower score)
            dur_penalty = W_DUR * d
            
            score = W_RW * rw + urgency - load_penalty - dur_penalty

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)

# EVOLVE-BLOCK-END
