"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity. Higher score is better.
    The item goes to the highest-scoring bin.
    
    Uses First Fit heuristic:
    1. Identify valid bins (remaining >= item).
    2. Among valid bins, select the one with the lowest index.
    3. This is implemented by assigning scores such that lower indices have higher scores.
    
    First Fit is a strong baseline for online bin packing and may outperform Best Fit
    on certain distributions by preserving the structure of the bin list.
    """
    # Calculate remaining capacity
    remaining = bins - item
    
    # Filter to only bins that can hold the item
    valid_mask = remaining >= 0
    
    # Create score array with -inf for invalid bins
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    if not np.any(valid_mask):
        return scores
    
    # Get indices of valid bins
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    
    if n_valid == 0:
        return scores
    
    # Assign scores: First Fit prefers the lowest index.
    # Score for index i (among valid ones) is -i.
    # So valid_indices[0] gets 0, valid_indices[1] gets -1, etc.
    # This ensures the first valid bin (lowest original index) has the highest score.
    
    # We can vectorize this:
    # The rank of each valid index in the sorted list of valid indices is its position.
    # Since valid_indices is already sorted (np.where returns sorted indices),
    # the rank is just the position in the array.
    
    ranks = np.arange(n_valid)
    valid_scores = -ranks
    
    scores[valid_indices] = valid_scores
    
    return scores


# EVOLVE-BLOCK-END
