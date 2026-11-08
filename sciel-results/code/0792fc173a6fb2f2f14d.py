"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global deadline guard
_START_TIME = _time.time()
_HARD_LIMIT = 110.0  # seconds
_SOFT_MARGIN = 5.0   # seconds

def _check_deadline():
    """Return True if we are approaching the hard limit."""
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Composite heuristic blending LPT (Longest Processing Time)
    with Most Remaining Work (MRW). In Open-Shop scheduling, LPT is a strong
    heuristic for minimizing makespan by keeping machines busy. Adding a
    weighted component for remaining work helps balance job completion and
    machine utilization, potentially reducing suboptimal sequencing.
    
    We use the minimum processing time among available machine options as the
    effective duration of the operation.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Check deadline - if too close, fall back to a very fast rule
    if _check_deadline():
        # Fallback: Just pick the one with max remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Calculate the effective processing time for each operation
    # We assume the 'cost' of the operation is the minimum duration available
    # in its machine options. This represents the shortest time the operation
    # could take if we had the best machine available.
    
    min_durations = np.zeros(n)
    remaining_works = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if machine_options:
            durations = list(machine_options.values())
            min_durations[i] = min(durations)
        else:
            min_durations[i] = 0.0
        remaining_works[i] = op["remaining_work"]
    
    # Normalize both criteria to [0, 1] range for weighted combination
    dur_range = np.ptp(min_durations)
    work_range = np.ptp(remaining_works)
    
    if dur_range > 0:
        norm_durations = (min_durations - min_durations.min()) / dur_range
    else:
        norm_durations = np.zeros(n)
    
    if work_range > 0:
        norm_works = (remaining_works - remaining_works.min()) / work_range
    else:
        norm_works = np.zeros(n)
    
    # Weighted composite score: 70% LPT, 30% MRW
    # This prioritizes long operations while still considering job progress
    weights = np.array([0.7, 0.3])
    composite_scores = weights[0] * norm_durations + weights[1] * norm_works
    
    # Select the operation with the highest composite score
    best_idx = int(np.argmax(composite_scores))
    
    return best_idx
# EVOLVE-BLOCK-END
