"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Global state for adaptive scoring
_item_count = 0
_item_sum = 0.0
_mean_item_size = 50.0  # Initial guess, will be updated

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Adaptive Mean-Targeting.
    We model the bin packing as an inventory system. We estimate the "typical"
    future item size (using the running mean of observed items). We score each
    bin based on how close its remaining capacity (after placing the current item)
    is to this typical size. This steers the system towards creating bins with
    remainders that are likely to be filled by the next item, minimizing
    fragmentation and waste.
    
    Score = - (remaining_after - mean_item_size)^2
    """
    global _item_count, _item_sum, _mean_item_size

    # Check if we're close to the deadline
    if time.time() - _START_TIME > _MAX_EVAL_TIME:
        # Emergency: return a simple fallback (Best-Fit)
        remaining = bins - item
        scores = np.where(remaining >= 0, -remaining, -np.inf)
        return scores

    # Update running statistics
    _item_count += 1
    _item_sum += item
    if _item_count > 0:
        _mean_item_size = _item_sum / _item_count
    
    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        return np.full(len(bins), -np.inf)
    
    scores = np.full(len(bins), -np.inf)
    
    # Calculate score for feasible bins
    # We want the remainder to be close to the mean item size.
    # If the remainder is close to the mean, the next item (which is likely near the mean)
    # will fit well, leaving minimal waste.
    # If the remainder is much smaller than the mean, the next item likely won't fit.
    # If the remainder is much larger than the mean, we waste space.
    
    # Use the running mean as the target
    target = _mean_item_size
    
    # To avoid issues with very early items where the mean is noisy,
    # we can blend with a default or just use it.
    # For robustness, if count is very low, we might prefer Best-Fit.
    if _item_count < 10:
        # Fallback to Best-Fit for the first few items to establish a baseline
        scores[feasible] = -remaining_after[feasible]
    else:
        # Adaptive scoring
        diffs = remaining_after[feasible] - target
        scores[feasible] = -(diffs ** 2)
        
    return scores
# EVOLVE-BLOCK-END
