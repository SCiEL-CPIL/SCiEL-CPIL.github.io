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
        
    # Quadratic Best Fit: Strongly penalize large remaining capacity
    # This focuses even more on minimizing fragmentation by preferring the tightest fit
    # score = -(bins - item)^2
    waste = bins - item
    scores = -waste**2
    
    # Mask invalid bins
    scores[~valid] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
