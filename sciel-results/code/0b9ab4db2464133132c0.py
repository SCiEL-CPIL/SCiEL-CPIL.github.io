"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred.
    
    Strategy: Hybrid Best-Fit/Worst-Fit
    - Large items (> 25% of capacity): Use Worst Fit (maximize remaining space)
      to keep bins open for other large items.
    - Small items (<= 25% of capacity): Use Best Fit (minimize remaining space)
      to consolidate small items and reduce fragmentation.
    - Tie-breaking: Prefer lower indices (First Fit emulation).
    """
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores
    
    n = len(bins)
    scores = np.full(n, -1e18)
    
    # Calculate remaining capacity after placement
    remaining = bins - item
    
    # Only consider bins where remaining >= 0
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_remaining = remaining[valid_mask]
    
    # Threshold for "large" items. Capacity is 100.
    # Weibull distributions are skewed; 25 is a reasonable cutoff for "large".
    LARGE_ITEM_THRESHOLD = 25.0
    
    # Index penalty to break ties in favor of earlier bins (First Fit behavior)
    index_penalty = valid_indices * 0.1
    
    if item > LARGE_ITEM_THRESHOLD:
        # Worst Fit: Maximize remaining space
        # Higher remaining is better.
        scores[valid_indices] = valid_remaining - index_penalty
    else:
        # Best Fit: Minimize remaining space
        # Lower remaining is better.
        scores[valid_indices] = -valid_remaining - index_penalty
    
    return scores
# EVOLVE-BLOCK-END
