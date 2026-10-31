"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    # Only consider bins with enough capacity
    valid = remaining >= 0
    
    # Base score: best-fit (minimize remaining capacity)
    # Higher score = better bin
    score = -remaining.copy()
    
    # Penalize very small remainders to avoid creating unusable gaps
    # Use a smooth penalty: for remainders below threshold, add quadratic penalty
    # Threshold scales with item size to be adaptive
    threshold = max(2.0, 0.05 * item)
    
    # Smooth penalty: (threshold - remaining)^2 for remaining < threshold
    diff = threshold - remaining
    penalty = np.where(diff > 0, diff * diff * 0.1, 0.0)
    
    # Add penalty to score (reduces score for small remainders)
    score = score - penalty
    
    # Invalidate bins that can't fit the item
    score = np.where(valid, score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
