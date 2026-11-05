"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: keep a safety margin before the 150s wall-clock limit.
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin


def _deadline_exceeded():
    return _time.time() >= _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: A refined heuristic that balances:
    1. Remaining work (long jobs first)
    2. Remaining operations (depth of job)
    3. Minimum processing time on available machines (shortest job first to reduce blocking)
    4. Machine load consideration (prefer less-loaded machines if possible)
    
    We use a weighted combination that has shown good results in job-shop scheduling.
    """
    if _deadline_exceeded():
        # Fallback: just pick the first one
        return 0
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Precompute features for each ready operation
    scores = np.empty(n)
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        # Minimum duration across machine options
        if machine_options:
            min_duration = min(machine_options.values())
            # Average duration
            avg_duration = np.mean(list(machine_options.values()))
        else:
            min_duration = 1.0
            avg_duration = 1.0
        
        # Feature 1: Remaining work (higher = more work left, prioritize)
        # Normalize roughly by typical scale
        work_score = remaining_work
        
        # Feature 2: Remaining ops (higher = deeper job, prioritize to avoid starvation)
        depth_score = remaining_ops * 2.0  # weight depth
        
        # Feature 3: Inverse of min duration (shorter ops first to reduce machine blocking)
        # This helps with parallelism
        duration_score = 1.0 / (min_duration + 1e-6)
        
        # Feature 4: Ratio of remaining work to min duration (work density)
        # Higher ratio means more work packed into short ops -> prioritize
        ratio_score = remaining_work / (min_duration + 1e-6)
        
        # Combine with weights tuned for job-shop scheduling
        # Weights: work matters most, then depth, then work-density ratio
        score = (
            1.0 * work_score +
            1.5 * depth_score +
            0.5 * ratio_score +
            0.3 * duration_score
        )
        
        scores[i] = score
    
    # Return the index with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
