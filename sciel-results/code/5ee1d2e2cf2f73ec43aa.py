"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin using a Worst Fit heuristic with power-law transformation.
    
    This function uses a power-law transformation of the remaining capacity to
    score bins. Unlike Best Fit, which minimizes remaining space, this heuristic
    maximizes the remaining space (Worst Fit) to preserve large contiguous blocks
    of space for potentially large future items. This is particularly useful for
    skewed distributions like Weibull where large items may arrive later.
    
    The score is higher for bins that leave more remaining space after placing the item.
    The exponent alpha controls the sensitivity.
    """
    start_time = time.time()
    
    # Capacity is fixed at 100 per task description
    capacity = 100.0
    
    # Compute remaining capacity if item is placed in each bin
    remaining = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Create scores array, default to very low for invalid bins
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    # Normalize remaining capacity to [0, 1]
    # Ensure we don't divide by zero
    u = remaining[valid_mask] / capacity
    
    # Worst Fit Heuristic
    # Score is proportional to the remaining capacity.
    # Higher score means more space left, which we prefer to keep bins "open".
    # alpha = 1.0 corresponds to linear Worst Fit.
    # alpha < 1.0 flattens the curve, making it less aggressive in choosing the absolute largest space.
    # We start with alpha = 1.0 as a baseline for Worst Fit.
    alpha = 1.0
    
    # Score: u^alpha
    # Larger u (looser fit) -> higher score
    # Smaller u (tighter fit) -> lower score
    scores[valid_mask] = u ** alpha
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 140:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
