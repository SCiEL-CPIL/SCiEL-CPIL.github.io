"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin for placing the given item.
    
    Strategy: Best Fit with fragmentation penalty.
    - Base score: minimize remaining space after placement (Best Fit)
    - Penalty: if remaining space is very small (< 10% of capacity), it's likely wasted
      and won't fit future items, so penalize such bins slightly
    - This avoids creating many tiny fragments that can't be used
    """
    # Filter bins with enough capacity
    valid = bins >= item
    if not np.any(valid):
        return np.full_like(bins, -np.inf, dtype=float)
    
    # Calculate remaining capacity after placing item
    remaining = bins - item
    
    # Base score: Best Fit (prefer smallest remaining)
    # We want to maximize score, so use negative remaining
    scores = -remaining.copy()
    
    # Fragmentation penalty: if remaining is very small, it's likely wasted
    # For Weibull distributions, small items are common, but extremely small
    # fragments (< 5-10 units for capacity 100) are hard to use
    threshold = 8.0  # 8% of capacity 100
    penalty_weight = 0.3
    
    # Apply penalty to bins that would leave very small remaining space
    small_remaining = (remaining < threshold) & valid
    scores[small_remaining] -= penalty_weight * (threshold - remaining[small_remaining])
    
    # Invalidate bins that can't fit the item
    scores[~valid] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
