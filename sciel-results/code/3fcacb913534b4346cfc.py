"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Hybrid strategy:
    # - For large items (item > 50), use Worst Fit to keep bins open for other large items
    # - For small items (item <= 50), use Best Fit to consolidate and reduce fragmentation
    # 
    # This addresses the likely cause of poor performance: the parent used Best Fit
    # for all items, which fragments bins when large items arrive later.
    
    n_bins = len(bins)
    scores = np.full(n_bins, -np.inf)
    
    if item > 50.0:
        # Worst Fit: prefer bins with the most remaining space
        # This keeps bins with lots of space available for future large items
        scores[valid] = bins[valid]
    else:
        # Best Fit: prefer bins with the least remaining space (that can fit the item)
        # This consolidates small items and reduces the number of partially-filled bins
        scores[valid] = -(bins[valid] - item)
    
    return scores
# EVOLVE-BLOCK-END
