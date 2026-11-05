"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Deadline guard: 140s is a safe margin before the 150s kill limit.
_START_TIME = _time.time()
_DEADLINE = 140.0


def dispatch(ready_operations, time):
    """
    Dispatch rule: Weighted combination of job urgency (remaining work per op)
    and normalized processing time (SPT relative to job's remaining work).
    
    Strategy:
    1. For each ready operation, compute the minimum processing time across all
       machine options (shortest processing time, SPT).
    2. Compute job urgency as remaining_work / remaining_ops (average workload per op).
    3. Compute a normalized SPT penalty: min_duration / remaining_work.
       This makes the penalty relative to the job's total remaining work,
       so critical jobs with high remaining work get a smaller relative penalty.
    4. Select the operation with the highest combined score:
       score = urgency - alpha * normalized_spt_penalty
    
    The urgency term captures how much work the job still has (critical path),
    while the normalized SPT penalty prefers shorter operations without
    over-penalizing critical jobs that need to keep moving.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Check deadline
    if _time.time() > _DEADLINE:
        # Fallback to simple rule: max remaining work
        return int(max(range(n), key=lambda i: ready_operations[i]["remaining_work"]))
    
    if n == 1:
        return 0
    
    best_idx = 0
    best_score = -float('inf')
    best_min_duration = float('inf')
    
    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        min_duration = min(durations) if durations else 0
        
        remaining_work = op["remaining_work"]
        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1
        
        # Urgency: average remaining work per operation
        urgency = remaining_work / remaining_ops
        
        # Normalized SPT penalty: relative to job's remaining work
        if remaining_work > 0:
            nspt_penalty = min_duration / remaining_work
        else:
            nspt_penalty = min_duration
        
        # Combined score: urgency dominates, NSPT refines
        # Weight of 0.3 for the SPT penalty balances urgency with processing time
        score = urgency - 0.3 * nspt_penalty
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_min_duration = min_duration
        elif score == best_score:
            # Tie-breaker: Prefer shorter processing time
            if min_duration < best_min_duration:
                best_idx = i
                best_min_duration = min_duration
                
    return int(best_idx)
# EVOLVE-BLOCK-END
