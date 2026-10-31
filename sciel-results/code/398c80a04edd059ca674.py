"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    valid = remaining >= 0
    
    # Best Fit Decreasing style: minimize the remaining capacity after placement
    # This is equivalent to maximizing the fill ratio of the bin
    # Score = -remaining (lower remaining is better, so we negate for max)
    
    # Only consider valid bins
    score = np.where(valid, -remaining, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
