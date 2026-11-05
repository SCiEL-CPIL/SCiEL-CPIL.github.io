"""Job-shop dispatch rule: Critical Path (LRPT) with smooth machine-contention."""
# EVOLVE-BLOCK-START
import time as _time
import math

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 140.0  # safety margin well before the 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    1. Dominant term: Remaining Work (LRPT) to clear critical paths.
    2. Smooth machine-contention penalty via log-sum-exp soft-min to avoid hard jumps.
    3. Urgency bonus for jobs with fewer remaining operations.
    4. Slight SPT preference for shorter durations.
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
    W_RW = 1.0          # Weight for Remaining Work (LRPT)
    W_LOAD = 2.5        # Weight for Machine Load Penalty (increased for smoother contention)
    W_DUR = 0.1         # Weight for Duration (SPT) - slightly increased
    W_URGENCY = 0.5     # Weight for Urgency (1/remaining_ops) - slightly reduced

    best_idx = 0
    best_score = float("-inf")

    # For smooth contention, we use a log-sum-exp based penalty
    # This avoids the hard jump when load changes by 1
    # We compute a soft penalty that is smooth in the load values
    
    # First pass: compute raw scores without load penalty
    raw_scores = []
    for i in range(n):
        op = ready_operations[i]
        rw = float(op.get("remaining_work", 0) or 0)
        rem_ops = int(op.get("remaining_ops", 1) or 1)
        
        if rem_ops < 1:
            rem_ops = 1

        m = best_machines[i]
        d = best_durations[i]
        
        if m is None:
            # Fallback if no machine options
            raw_scores.append(rw)
        else:
            d = max(d, 0.1)  # Avoid division by zero
            urgency = W_URGENCY / rem_ops
            dur_penalty = W_DUR * d
            raw_scores.append(W_RW * rw + urgency - dur_penalty)

    # Second pass: compute smooth load penalty
    # For each operation, the penalty is based on the load of its best machine
    # We use a smooth approximation: penalty = W_LOAD * load / (max_load + 1)
    # This is already smooth, but let's use a log-sum-exp based approach for better behavior
    
    # Actually, a simpler smooth approach: use the normalized load with a sigmoid-like scaling
    # penalty = W_LOAD * (load / max_load) ** 0.8  # Slightly sublinear to reduce harsh penalties
    
    for i in range(n):
        op = ready_operations[i]
        m = best_machines[i]
        
        if m is None:
            score = raw_scores[i]
        else:
            load = machine_loads.get(m, 0)
            norm_load = load / max_load
            # Use a smooth penalty that is sublinear to avoid over-penalizing
            # This helps when multiple operations compete for the same machine
            load_penalty = W_LOAD * (norm_load ** 0.7)
            score = raw_scores[i] - load_penalty

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)

# EVOLVE-BLOCK-END
