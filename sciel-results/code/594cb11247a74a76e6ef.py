"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 120.0
_START_TIME = None

def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.monotonic()

def _elapsed():
    _ensure_start()
    return _time.monotonic() - _START_TIME

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: We want to minimize makespan. In open-shop scheduling, a good heuristic
    is to prioritize operations that:
    1. Have the most remaining work on the job (to complete long jobs early)
    2. Have the shortest processing time on their assigned machine (to free up machines quickly)
    3. Consider machine load - prefer machines that are less loaded
    
    We use a weighted combination:
    - Primary: remaining_work (more is better for job completion)
    - Secondary: minimum duration among machine_options (shorter is better for machine utilization)
    - Tertiary: remaining_ops (fewer remaining ops means the job is closer to completion)
    
    We'll compute a score for each ready operation and pick the one with the best (lowest) score.
    """
    _ensure_start()
    
    if not ready_operations:
        return 0
    
    # Check deadline with safety margin
    if _elapsed() > _DEADLINE - 5.0:
        # Fallback to simple rule if we're running out of time
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    
    # Compute features for each ready operation
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # For each operation, find the minimum duration among its machine options
    # This represents the fastest we can process this operation
    min_durations = np.zeros(n, dtype=float)
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if machine_options:
            min_durations[i] = min(machine_options.values())
        else:
            min_durations[i] = 0.0
    
    # Normalize features to [0, 1] range for consistent weighting
    # remaining_work: higher is better (we want to process jobs with more work first)
    if np.max(remaining_works) > np.min(remaining_works):
        rw_norm = (remaining_works - np.min(remaining_works)) / (np.max(remaining_works) - np.min(remaining_works))
    else:
        rw_norm = np.zeros(n)
    
    # min_duration: lower is better (we want to process operations that take less time first)
    if np.max(min_durations) > np.min(min_durations):
        md_norm = 1.0 - (min_durations - np.min(min_durations)) / (np.max(min_durations) - np.min(min_durations))
    else:
        md_norm = np.ones(n)
    
    # remaining_ops: lower is better (jobs with fewer remaining ops are closer to completion)
    if np.max(remaining_ops) > np.min(remaining_ops):
        ro_norm = 1.0 - (remaining_ops - np.min(remaining_ops)) / (np.max(remaining_ops) - np.min(remaining_ops))
    else:
        ro_norm = np.ones(n)
    
    # Weighted combination: emphasize remaining work most, then duration, then remaining ops
    # Weights: 0.5 for remaining work, 0.3 for min duration, 0.2 for remaining ops
    scores = 0.5 * rw_norm + 0.3 * md_norm + 0.2 * ro_norm
    
    # Pick the operation with the highest score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
