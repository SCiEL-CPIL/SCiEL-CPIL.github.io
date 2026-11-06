"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing using rank-based Best-Fit.
    
    Strategy: Rank-based scoring that favors bins with the smallest remaining
    capacity after placement, using the count of bins with smaller remainders
    as the score. This distribution-aware approach better handles the shape
    of the remainder distribution and reduces fragmentation by preferring
    gaps in the lower tail.
    
    A small bonus is added for bins that are nearly full to encourage closing.
    """
    start_time = time.time()
    
    # Calculate remaining capacity for each bin
    remaining = bins - item
    
    # Filter: only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    n_valid = len(valid_remaining)
    
    # Rank-based scoring: score = -count of valid bins with smaller remaining
    # This gives the tightest fit the highest score (most bins have larger remaining)
    # and the loosest fit the lowest score (fewest bins have smaller remaining)
    # Using argsort-based ranking for O(n log n) performance
    
    # For each valid bin, count how many valid bins have strictly smaller remaining
    # score[i] = -count of j where valid_remaining[j] < valid_remaining[i]
    # This is equivalent to the negative of the rank (0-indexed) in sorted order
    
    # Efficient computation using searchsorted on sorted remainders
    sorted_rem = np.sort(valid_remaining)
    
    # For each valid_remaining value, find its rank (number of elements < it)
    # searchsorted with side='left' gives the index where the element would be
    # inserted, which equals the count of elements strictly less than it
    ranks = np.searchsorted(sorted_rem, valid_remaining, side='left')
    
    # Score: negative rank (lower rank = tighter fit = higher score)
    base_score = -ranks.astype(float)
    
    # Small bonus for bins that are almost full (remaining < 5% of capacity)
    # This encourages completing bins to reduce the count of open bins
    threshold = 5.0
    bonus = 0.05 * (valid_remaining < threshold).astype(float)
    
    final_scores = base_score + bonus
    
    scores[valid_mask] = final_scores
    
    # Safety check for time
    if time.time() - start_time > 1.0:
        pass  # Return whatever we have, which is already computed
        
    return scores
# EVOLVE-BLOCK-END
