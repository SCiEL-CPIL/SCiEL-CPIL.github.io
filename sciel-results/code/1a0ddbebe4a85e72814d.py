"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Uses NSPT (Normalized Shortest Processing Time) strategy:
    - For each ready operation, find the minimum duration across its machine options
    - Normalize by remaining_work to balance machine-freeing (SPT) with job-progress (MWRT)
    - Prefer operations with high (min_duration / remaining_work) ratio
    - This avoids starvation of long jobs while still preferring quick ops among comparable jobs
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    if n == 1:
        return 0
    
    # Deadline guard: track elapsed time to avoid timeout
    # The evaluator kills after 150s, so we use a tight internal budget per call
    # Since this is a single dispatch call, it should be fast, but we add safety
    start = _time.perf_counter()
    deadline = start + 0.05  # 50ms max for this dispatch call
    
    scores = np.empty(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        # Check deadline
        if _time.perf_counter() > deadline:
            # Fallback: return best score found so far, or 0
            # Fill remaining with -inf so we pick from what we have
            scores[i:] = -np.inf
            break
        
        machine_options = op.get("machine_options", {})
        if not machine_options:
            scores[i] = 0.0
            continue
        
        # Minimum duration across machine options (SPT component)
        min_dur = min(machine_options.values())
        
        # Remaining work (MWRT component)
        rem_work = op.get("remaining_work", 1.0)
        if rem_work <= 0:
            rem_work = 1.0
        
        # NSPT: normalize min_duration by remaining_work
        # Higher means: this op is short relative to the job's total remaining work
        # This prefers quick ops for jobs that still have a lot of work (progress guarantee)
        # while also preferring quick ops overall (machine-freeing)
        scores[i] = min_dur / rem_work
    
    # Add a small tie-breaking: among equal scores, prefer the job with most remaining ops
    # to avoid idle time on machines that will be needed soon
    # Actually, let's keep it simple: argmax on NSPT score
    # But to break ties deterministically and favor progress, add tiny epsilon based on remaining_ops
    remaining_ops_arr = np.array([op.get("remaining_ops", 0) for op in ready_operations], dtype=np.float64)
    
    # Combined score: NSPT + small weight on remaining_ops to break ties favoring jobs with more work
    # This helps because a job with more remaining ops benefits more from starting now
    combined = scores + 1e-6 * remaining_ops_arr
    
    best_idx = int(np.argmax(combined))
    
    return best_idx
# EVOLVE-BLOCK-END
