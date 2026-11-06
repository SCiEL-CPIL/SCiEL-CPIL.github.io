"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for Job-Shop Scheduling.
    
    Uses a composite priority score that balances:
    1. Shortest processing time (min duration across machines) - reduces immediate blocking
    2. Remaining work - favors jobs closer to completion
    3. Remaining operations count - favors jobs with fewer ops left
    
    The score combines these to pick the operation that best reduces makespan.
    """
    if not ready_operations:
        return 0
    
    # Deadline guard: if we're getting close to the limit, use fast fallback
    if _time.time() > _DEADLINE:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    
    # Compute metrics for each ready operation
    min_durations = []
    remaining_works = []
    remaining_ops = []
    
    for op in ready_operations:
        # Minimum duration across all machine options
        min_dur = min(op["machine_options"].values())
        min_durations.append(min_dur)
        
        # Remaining total work
        remaining_works.append(op["remaining_work"])
        
        # Number of remaining operations
        remaining_ops.append(op["remaining_ops"])
    
    # Convert to numpy arrays for vectorized computation
    min_durations = np.array(min_durations, dtype=np.float64)
    remaining_works = np.array(remaining_works, dtype=np.float64)
    remaining_ops = np.array(remaining_ops, dtype=np.float64)
    
    # Normalize each metric to [0, 1] range for fair combination
    # For min_durations: lower is better, so we invert
    # For remaining_works: lower is better (closer to done)
    # For remaining_ops: lower is better
    
    def normalize(arr, reverse=False):
        """Normalize array to [0,1]. If reverse, lower values get higher scores."""
        if len(arr) == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        normed = (arr - mn) / (mx - mn)
        if reverse:
            normed = 1.0 - normed
        return normed
    
    # Higher score = better priority
    score_min_dur = normalize(min_durations, reverse=True)  # Shorter is better
    score_rem_work = normalize(remaining_works, reverse=True)  # Less work remaining is better
    score_rem_ops = normalize(remaining_ops, reverse=True)  # Fewer ops remaining is better
    
    # Weighted combination - emphasize shortest processing time most
    # SPT is known to be effective for job-shop scheduling
    weights = np.array([0.5, 0.3, 0.2])
    total_score = weights[0] * score_min_dur + weights[1] * score_rem_work + weights[2] * score_rem_ops
    
    # Return index of operation with highest priority score
    return int(np.argmax(total_score))
# EVOLVE-BLOCK-END
