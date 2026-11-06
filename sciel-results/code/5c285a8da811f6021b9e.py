"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for adaptive parameters
_start_time = time.time()
_deadline = 150.0  # seconds

# We'll use a simple heuristic that combines best-fit with a gap penalty
# For online bin packing, a good heuristic is to consider:
# 1. Best-fit: minimize remaining space (good for tight packing)
# 2. But avoid creating unusable gaps
#
# Score = -(remaining) + lambda * f(remaining/item)
# where f penalizes ratios that are likely to create wasted space

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    
    Uses Relative Best-Fit: minimizes (remaining / item).
    This normalizes the gap by the item size, potentially improving
    packing efficiency by avoiding gaps that are disproportionately large
    relative to the current item.
    """
    remaining = bins - item
    valid = remaining >= 0
    
    if not np.any(valid):
        return np.full(len(bins), -np.inf)
    
    # Relative Best-Fit score: minimize remaining/item
    # Equivalent to maximizing -remaining/item
    with np.errstate(divide='ignore', invalid='ignore'):
        score = np.where(valid, -remaining / item, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
