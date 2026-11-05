"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Deadline guard: 140s is a safe margin before the 150s kill limit.
_START_TIME = _time.time()
_DEADLINE = 140.0


def dispatch(ready_operations, time):
    """
    Dispatch rule: Multi-criteria combining Critical Ratio (CR), urgency, and SPT.
    
    Strategy:
    1. For each ready operation, compute the minimum processing time across all
       machine options (shortest processing time, SPT).
    2. Estimate slack time as remaining_work - current_time (proxy for deadline pressure).
    3. Compute Critical Ratio: slack / min_duration. Lower CR = more urgent.
    4. Compute job urgency as remaining_work / remaining_ops (average workload per op).
    5. Compute normalized SPT penalty: min_duration / remaining_work.
    6. Select the operation with the highest combined score:
       score = -beta * CR + alpha * urgency - gamma * normalized_spt_penalty
    
    The CR term captures dynamic deadline pressure, urgency captures static workload,
    and SPT prefers shorter operations to reduce machine contention.
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
    
    # Weights: CR is primary (deadline pressure), urgency secondary, SPT tertiary
    beta = 1.0   # CR weight (negative: lower CR is better)
    alpha = 0.5  # Urgency weight
    gamma = 0.2  # SPT penalty weight
    
    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        min_duration = min(durations) if durations else 0
        
        remaining_work = op["remaining_work"]
        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1
        
        # Critical Ratio: slack / processing time
        # Slack = remaining_work - time (estimate of time left before "deadline")
        slack = remaining_work - time
        if min_duration > 0:
            cr = slack / min_duration
        else:
            cr = 0.0 if slack >= 0 else -1.0
        
        # Urgency: average remaining work per operation
        urgency = remaining_work / remaining_ops
        
        # Normalized SPT penalty: relative to job's remaining work
        if remaining_work > 0:
            nspt_penalty = min_duration / remaining_work
        else:
            nspt_penalty = min_duration
        
        # Combined score: CR dominates, urgency and SPT refine
        score = -beta * cr + alpha * urgency - gamma * nspt_penalty
        
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
