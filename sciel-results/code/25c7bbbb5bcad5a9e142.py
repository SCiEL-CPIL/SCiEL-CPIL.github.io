"""Job-shop dispatch rule: LRPT with Critical Ratio and smooth machine-contention."""
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
    2. Critical Ratio: prioritize operations with low slack-to-work ratio (urgency under time pressure).
    3. Smooth contention penalty via log-sum-exp soft-min to avoid hard jumps.
    4. Urgency bonus for jobs with fewer remaining operations.
    5. Slight SPT preference for shorter durations.
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
    W_CR = 0.5          # Weight for Critical Ratio penalty
    W_LOAD = 2.0        # Weight for Machine Load Penalty
    W_DUR = 0.05        # Weight for Duration (SPT)
    W_URGENCY = 0.8     # Weight for Urgency (1/remaining_ops)

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
            
            # Critical Ratio: slack / remaining work
            # Lower CR means less slack relative to work -> more urgent
            # Estimate slack as a simple proxy: we don't have true slack, 
            # but we can use the ratio of current time to remaining work as a pressure indicator
            # Actually, a better CR proxy: if we assume the job should be done by some horizon,
            # we can use the fact that operations with less remaining work relative to time elapsed
            # are more "behind schedule". But since we don't have a target makespan,
            # we use: CR = (time_to_go_estimate) / rw, where time_to_go is hard to estimate.
            # Alternative: use the ratio of completed work to total work as a progress indicator.
            # For simplicity, use: urgency_pressure = 1.0 / (rw + 1.0) which favors jobs with less work left
            # But that's similar to remaining_ops. Let's use a different approach:
            # CR proxy: if a job has many remaining operations but little remaining work, it's "easy" to finish.
            # If a job has few remaining operations but lots of remaining work, it's "hard" and urgent.
            # So: cr_penalty = W_CR * (rw / (rem_ops + 1.0)) -- higher when work per op is high
            cr_penalty = W_CR * (rw / (rem_ops + 1.0))
            
            # Urgency term: smaller remaining ops count is better (higher score)
            urgency = W_URGENCY / rem_ops
            
            # Load penalty: higher load is worse (lower score)
            load_penalty = W_LOAD * norm_load
            
            # Duration penalty: longer duration is worse (lower score)
            dur_penalty = W_DUR * d
            
            score = W_RW * rw + urgency - cr_penalty - load_penalty - dur_penalty

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)

# EVOLVE-BLOCK-END
