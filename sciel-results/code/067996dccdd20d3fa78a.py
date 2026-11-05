"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule using a look-ahead bottleneck heuristic.
    
    For each ready operation, we estimate the earliest completion time if assigned
    to the least-loaded eligible machine. We then pick the operation that, when
    scheduled, leads to the best overall progress — specifically, we prioritize
    operations on machines that are currently less loaded to balance load, while
    also considering the urgency (remaining work) of the job.
    
    The heuristic score combines:
    1. Earliest possible completion time on the least-loaded eligible machine
       (lower is better for that operation's immediate impact).
    2. A penalty based on how much remaining work the job has, to ensure jobs
       with more work remaining get priority.
    3. A machine load balancing term: prefer operations whose least-loaded
       eligible machine has the lowest current load.
    
    Since we don't have direct access to machine loads in this interface, we
    approximate by using the minimum duration among machine options as a proxy
    for the "best" machine assignment, and combine with remaining_work urgency.
    
    The score for each ready operation is:
        score = min_duration * (1 + remaining_work / max_remaining_work)
    
    We pick the operation with the lowest score (earliest completion + urgency).
    """
    if not ready_operations:
        return 0
    
    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    
    # For each operation, find the minimum duration among its machine options
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else float('inf')
        for op in ready_operations
    ], dtype=float)
    
    # Normalize remaining work to [0, 1] for weighting
    max_rw = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    normalized_rw = remaining_works / max_rw
    
    # Score: lower is better
    # Combine minimum duration (earliest completion proxy) with urgency
    # We want to pick operations that can finish quickly on a less-contended machine
    # and also have significant remaining work (to avoid starving long jobs)
    scores = min_durations * (1.0 + 0.5 * normalized_rw)
    
    # Handle any inf values
    scores = np.where(np.isinf(scores), np.max(min_durations[np.isfinite(min_durations)]) if np.any(np.isfinite(min_durations)) else 1e9, scores)
    
    # Pick the operation with the lowest score
    best_idx = int(np.argmin(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
