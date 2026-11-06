"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Filter bins with enough remaining capacity
    remaining = bins - item
    valid_mask = remaining >= 0
    
    # If no valid bins, return -inf for all (shouldn't happen per problem statement)
    if not np.any(valid_mask):
        return np.full_like(bins, -np.inf)
    
    # Rank-based scoring: for each valid bin, count how many other valid bins
    # have strictly smaller remaining capacity. Lower rank = tighter fit.
    # Score = -rank, so the tightest fit gets the highest score.
    # This is equivalent to best-fit but handles ties by index order implicitly
    # through the rank computation, and may better handle distribution shape.
    
    valid_remaining = remaining[valid_mask]
    n_valid = len(valid_remaining)
    
    # Compute rank: number of valid bins with strictly smaller remaining
    # For each valid bin i, rank_i = sum(valid_remaining < valid_remaining[i])
    # This is O(n^2) but n is typically small (number of open bins)
    # For speed, we can use searchsorted on sorted array
    sorted_rem = np.sort(valid_remaining)
    
    # Rank of each element: number of elements strictly less than it
    # Use searchsorted with side='left' to get count of elements < x
    ranks = np.searchsorted(sorted_rem, valid_remaining, side='left')
    
    # Score: higher is better, so we want tightest fit (smallest remaining) to have highest score
    # rank 0 = smallest remaining = best fit
    # score = -rank (so rank 0 -> 0, rank 1 -> -1, etc.)
    # But we want the highest score to be selected, so we invert: score = -ranks
    scores = np.full_like(bins, -np.inf)
    scores[valid_mask] = -ranks.astype(float)
    
    # Add a small tie-breaker: among bins with the same rank (same remaining capacity),
    # prefer the one with more total remaining capacity (fuller bin) to consolidate
    # This is a minor adjustment
    fill_ratio = bins / 100.0
    scores[valid_mask] += 0.001 * fill_ratio[valid_mask]
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen, but return best found
        pass
    
    return scores
# EVOLVE-BLOCK-END
