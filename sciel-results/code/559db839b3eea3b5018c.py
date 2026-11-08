"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Deadline guard: ensure we return well before the 120s (or 150s) kill.
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # 115s safety margin


def _deadline_exceeded():
    return _time.time() > _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Combine multiple heuristics to approximate a good dispatch rule:
    1. Minimum Remaining Work (MRW) - prefer jobs with less total remaining work
    2. Longest Processing Time (LPT) on the assigned machine - prefer shorter ops first? 
       Actually for makespan, we often want to balance. Let's use a composite score.
    
    For job-shop scheduling, common effective heuristics include:
    - SPT (Shortest Processing Time) on the machine
    - MRW (Minimum Remaining Work) for the job
    - Critical path based rules
    
    Let's use a weighted combination:
    - For each ready op, compute a score. Lower score = better (start first).
    - Score = alpha * (remaining_work / remaining_ops) + beta * min_duration_on_machine + gamma * (1/remaining_ops)
    
    Actually, a simpler and often effective approach for makespan minimization in job shops:
    - Prefer operations that are on the critical path.
    - Use a surrogate: score = min_duration + remaining_work * 0.1
    
    Let's try: score = min_duration + 0.1 * remaining_work
    Lower score means we should pick it.
    
    But wait, we want to minimize makespan. Starting operations that free up machines early or 
    that are on the critical path helps. 
    
    Alternative: Use the "most constrained job first" - jobs with fewer remaining ops or 
    machines with higher load.
    
    Let's use a simple but effective heuristic:
    - For each ready op, compute the minimum duration across its machine options.
    - Also consider the remaining work of the job.
    - Score = min_duration + 0.05 * remaining_work
    - Pick the op with the lowest score.
    
    This balances short processing time (to keep machines busy with quick tasks) 
    with considering the total remaining work (to avoid starving jobs with long tails).
    """
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        # Fallback: just pick the first one
        return 0
    
    n = len(ready_operations)
    scores = np.empty(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        # Minimum duration among machine options
        min_dur = min(op["machine_options"].values())
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Composite score: lower is better
        # We want to pick ops that have short processing times and jobs with less remaining work
        # to avoid creating long tails.
        # Also, consider the average remaining work per op to normalize.
        avg_remaining = remaining_work / max(remaining_ops, 1)
        
        # Weighted sum
        score = 1.0 * min_dur + 0.1 * avg_remaining
        
        scores[i] = score
    
    # Pick the operation with the lowest score
    best_idx = int(np.argmin(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
