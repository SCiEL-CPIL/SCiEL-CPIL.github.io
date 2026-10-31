"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Strategy: Relative Super-Best Fit.
    We treat the packing as a continuous fluid flow problem where we minimize
    the 'potential energy' associated with the remaining capacity relative to
    the item size. The score is based on the square of the ratio of remaining
    capacity to item size, penalizing large relative gaps more heavily than
    linear Best Fit. This prioritizes bins where the item constitutes a large
    fraction of the remaining capacity, minimizing the relative 'waste' of
    opening new bins.
    """
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback to simple Best Fit if time is running out
        return -(bins - item)

    remaining_after = bins - item
    feasible = remaining_after >= 0
    
    if not np.any(feasible):
        return np.full(len(bins), -np.inf)
        
    scores = np.full(len(bins), -np.inf)
    
    # Calculate scores for feasible bins
    rem = remaining_after[feasible]
    
    # Handle case where item is 0 to avoid division by zero
    if item > 0:
        # Ratio of remaining capacity to item size
        # We want to minimize this ratio.
        # Score is negative of the squared ratio.
        # Higher score (closer to 0) is better.
        scores[feasible] = -((rem / item) ** 2)
    else:
        # If item is 0, any bin is fine, but prefer smallest remaining (Best Fit)
        scores[feasible] = -rem
        
    return scores
# EVOLVE-BLOCK-END
