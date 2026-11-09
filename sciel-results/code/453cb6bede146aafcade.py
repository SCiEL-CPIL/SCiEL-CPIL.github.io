"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the item.
    The item goes to the highest-scoring bin.
    
    Approach: Residual Utility Heuristic
    - We estimate the expected utility of the remaining space after placing the item.
    - For Weibull-distributed items (shape ~1-2, scale chosen for capacity 100),
      small items are more common than large ones.
    - A residual is valuable if it can fit at least one more typical item.
    - We score based on: (1) whether residual can fit a "median" future item,
      (2) the number of items that could potentially fit in the residual,
      (3) penalty for residuals that are too small to be useful but not zero.
    """
    remaining = bins
    capacity = 100.0
    
    # Only consider bins that can fit the item
    mask = remaining >= item
    if not np.any(mask):
        return np.full_like(remaining, -np.inf)
    
    # Residual space after placing the item
    residual = remaining[mask] - item
    
    # Estimate typical item sizes for Weibull distribution with capacity 100.
    # For Weibull with shape k=1.5, scale lambda=30, mean ~ 25, median ~ 21.
    # We'll use a few reference sizes to evaluate residual utility.
    
    # Reference item sizes (typical, medium, large)
    ref_small = 10.0   # Small items (common)
    ref_medium = 25.0  # Medium items (typical)
    ref_large = 50.0   # Large items (less common)
    
    # Utility: count how many reference sizes can fit in the residual
    # Weighted by their frequency (small items are more common)
    utility = (
        2.0 * (residual >= ref_small).astype(float) +
        1.5 * (residual >= ref_medium).astype(float) +
        1.0 * (residual >= ref_large).astype(float)
    )
    
    # Penalize residuals that are "awkward": too small to fit a small item 
    # but not zero (wasted space that can't be used)
    awkward_penalty = np.where(
        (residual > 0.5) & (residual < ref_small),
        -1.0 * (ref_small - residual) / ref_small,  # Scale penalty by how close to ref_small
        0.0
    )
    
    # Bonus for near-perfect fits (residual very close to 0)
    perfect_fit_bonus = np.where(
        residual < 1.0,
        3.0 * (1.0 - residual),  # Strong bonus for near-zero residual
        0.0
    )
    
    # Combine: utility + bonus - penalty
    score = utility + perfect_fit_bonus + awkward_penalty
    
    # Map back to full array
    full_score = np.full_like(remaining, -np.inf)
    full_score[mask] = score
    
    return full_score
# EVOLVE-BLOCK-END
