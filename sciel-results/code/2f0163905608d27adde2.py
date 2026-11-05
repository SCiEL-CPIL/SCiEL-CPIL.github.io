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
    
    # First pass: compute raw features for normalization
    raw_work = np.empty(n)
    raw_depth = np.empty(n)
    raw_min_dur = np.empty(n)
    raw_avg_dur = np.empty(n)
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        if machine_options:
            min_duration = min(machine_options.values())
            avg_duration = np.mean(list(machine_options.values()))
        else:
            min_duration = 1.0
            avg_duration = 1.0
        
        raw_work[i] = remaining_work
        raw_depth[i] = remaining_ops
        raw_min_dur[i] = min_duration
        raw_avg_dur[i] = avg_duration
    
    # Normalize features to [0, 1] range for better comparability
    work_min, work_max = raw_work.min(), raw_work.max()
    depth_min, depth_max = raw_depth.min(), raw_depth.max()
    dur_min, dur_max = raw_min_dur.min(), raw_min_dur.max()
    
    # Avoid division by zero
    work_range = max(work_max - work_min, 1e-6)
    depth_range = max(depth_max - depth_min, 1e-6)
    dur_range = max(dur_max - dur_min, 1e-6)
    
    for i, op in enumerate(ready_operations):
        # Normalized features
        work_norm = (raw_work[i] - work_min) / work_range
        depth_norm = (raw_depth[i] - depth_min) / depth_range
        dur_norm = (raw_min_dur[i] - dur_min) / dur_range
        
        # Feature 1: Normalized remaining work (higher = more work left, prioritize)
        work_score = work_norm
        
        # Feature 2: Normalized remaining ops (higher = deeper job, prioritize to avoid starvation)
        depth_score = depth_norm
        
        # Feature 3: Inverse normalized duration (shorter ops get higher score)
        # This helps with parallelism and reduces machine blocking
        duration_score = 1.0 - dur_norm
        
        # Feature 4: Work-density ratio (remaining work / min duration)
        # Higher ratio means more work packed into short ops -> prioritize
        ratio_score = raw_work[i] / (raw_min_dur[i] + 1e-6)
        
        # Normalize ratio_score
        ratio_min = (raw_work / (raw_min_dur + 1e-6)).min()
        ratio_max = (raw_work / (raw_min_dur + 1e-6)).max()
        ratio_range = max(ratio_max - ratio_min, 1e-6)
        ratio_norm = (ratio_score - ratio_min) / ratio_range
        
        # Combine with refined weights tuned for job-shop scheduling
        # Emphasize work and depth more, with moderate weight on duration and ratio
        score = (
            1.2 * work_score +
            1.8 * depth_score +
            0.6 * duration_score +
            0.4 * ratio_norm
        )
        
        scores[i] = score
    
    # Return the index with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
