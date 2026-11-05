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
    
    Strategy: A hybrid ratio-based heuristic that combines:
    1. WSPT core: min_duration / (remaining_work + 1) -- primary signal
    2. Depth urgency: remaining_ops to avoid starving deep jobs
    3. Work urgency: remaining_work to prioritize long-tail jobs
    4. SPT benefit: shorter operations free machines faster
    
    The WSPT ratio is the dominant signal (favors short ops on jobs with
    lots of work left). Depth and work act as secondary refinements for
    tie-breaking and edge cases. SPT helps maintain parallelism.
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
    
    # First pass: compute raw features
    raw_work = np.empty(n)
    raw_depth = np.empty(n)
    raw_min_dur = np.empty(n)
    raw_ratio = np.empty(n)
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        if remaining_work is None or remaining_work < 0:
            remaining_work = 0.0
        remaining_ops = op.get("remaining_ops", 0)
        if remaining_ops is None or remaining_ops < 0:
            remaining_ops = 0
        machine_options = op.get("machine_options", {})
        
        if machine_options:
            durations = list(machine_options.values())
            min_duration = min(durations) if durations else 1.0
        else:
            min_duration = 1.0
        
        raw_work[i] = remaining_work
        raw_depth[i] = remaining_ops
        raw_min_dur[i] = min_duration
        # Core WSPT ratio: lower is better (shorter ops on jobs with more work)
        raw_ratio[i] = min_duration / (remaining_work + 1.0)
    
    # Normalize the ratio to [0, 1] (inverted so higher = better)
    ratio_min = raw_ratio.min()
    ratio_max = raw_ratio.max()
    ratio_range = max(ratio_max - ratio_min, 1e-6)
    
    # Normalize depth to [0, 1]
    depth_min = raw_depth.min()
    depth_max = raw_depth.max()
    depth_range = max(depth_max - depth_min, 1e-6)
    
    # Normalize work to [0, 1]
    work_min = raw_work.min()
    work_max = raw_work.max()
    work_range = max(work_max - work_min, 1e-6)
    
    # Normalize min_duration to [0, 1] (inverted so shorter = higher score)
    dur_min = raw_min_dur.min()
    dur_max = raw_min_dur.max()
    dur_range = max(dur_max - dur_min, 1e-6)
    
    for i in range(n):
        # Inverted normalized ratio: higher = shorter relative to work = better
        ratio_score = 1.0 - (raw_ratio[i] - ratio_min) / ratio_range
        
        # Normalized depth: higher = deeper job = prioritize to avoid starvation
        depth_score = (raw_depth[i] - depth_min) / depth_range
        
        # Normalized work: higher = more work left = prioritize
        work_score = (raw_work[i] - work_min) / work_range
        
        # Inverted normalized duration: higher = shorter op = better (SPT benefit)
        duration_score = 1.0 - (raw_min_dur[i] - dur_min) / dur_range
        
        # Combine: WSPT ratio is the dominant signal, with depth, work, and
        # duration as secondary refinements. The ratio already captures work
        # and duration, so we keep its weight high and use others for
        # tie-breaking and edge cases.
        score = (
            2.0 * ratio_score +
            1.2 * depth_score +
            0.6 * work_score +
            0.4 * duration_score
        )
        
        scores[i] = score
    
    # Return the index with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
