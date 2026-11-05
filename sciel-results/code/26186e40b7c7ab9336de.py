"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Deadline guard: track elapsed time to ensure we stay well within the 150s limit
_start_time = _time.time()
_DEADLINE = 140.0  # 10s margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Strategy: Combine multiple heuristics to select the best ready operation:
    1. Shortest Processing Time (SPT) - prioritize operations with shortest duration
    2. Most Work Remaining (MWR) - prioritize jobs with more remaining work
    3. Earliest Due Date consideration via remaining operations count
    
    We use a weighted combination that adapts based on the state of the schedule.
    """
    if not ready_operations:
        return 0
    
    # Deadline guard: if we're running out of time, use a fast fallback
    if _time.time() - _start_time > _DEADLINE:
        # Fast fallback: shortest processing time
        min_duration = float('inf')
        best_idx = 0
        for i, op in enumerate(ready_operations):
            min_opt = min(op["machine_options"].values())
            if min_opt < min_duration:
                min_duration = min_opt
                best_idx = i
        return int(best_idx)
    
    n = len(ready_operations)
    
    # Compute scores for each ready operation
    # We'll use a multi-criteria approach:
    # - Shortest processing time (on best available machine)
    # - Most remaining work (to keep busy jobs progressing)
    # - Fewest remaining operations (to finish jobs sooner)
    
    # First pass: compute features
    min_durations = []
    remaining_works = []
    remaining_ops = []
    
    for op in ready_operations:
        # Minimum duration across machine options
        min_dur = min(op["machine_options"].values())
        min_durations.append(min_dur)
        
        # Remaining work
        rw = op["remaining_work"]
        remaining_works.append(rw)
        
        # Number of remaining operations
        ro = op["remaining_ops"]
        remaining_ops.append(ro)
    
    # Normalize features to [0, 1] range
    def normalize(arr, reverse=False):
        mn = min(arr)
        mx = max(arr)
        if mx - mn < 1e-9:
            return [0.5] * len(arr)
        if reverse:
            return [(mx - x) / (mx - mn) for x in arr]
        return [(x - mn) / (mx - mn) for x in arr]
    
    # For SPT: lower is better, so we want to reverse-normalize (lower duration -> higher score)
    spt_scores = normalize(min_durations, reverse=True)
    
    # For MWR: higher remaining work is better (keeps jobs progressing)
    mwr_scores = normalize(remaining_works)
    
    # For fewest remaining ops: lower is better (finish jobs sooner)
    # Actually, we want to prioritize jobs with fewer remaining ops to complete them
    few_ops_scores = normalize(remaining_ops, reverse=True)
    
    # Weighted combination
    # SPT tends to reduce makespan by clearing short jobs quickly
    # MWR keeps busy jobs moving
    # Few ops helps complete jobs
    
    w_spt = 0.5
    w_mwr = 0.3
    w_ops = 0.2
    
    best_score = -1
    best_idx = 0
    for i in range(n):
        score = w_spt * spt_scores[i] + w_mwr * mwr_scores[i] + w_ops * few_ops_scores[i]
        if score > best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
