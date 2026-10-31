"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity using a probabilistic risk model
    based on minimizing the variance of the remaining capacity distribution.
    
    When an item is placed in a bin, that bin's remaining capacity decreases.
    We score each candidate bin by the negative of the resulting variance of
    the remaining capacities across all open bins. Bins whose placement leads
    to the lowest overall variance (most uniform distribution of remaining
    capacities) receive the highest scores, as this reduces fragmentation
    and improves asymptotic packing density.
    
    Returns an array of scores; the item goes to the highest-scoring bin.
    """
    remaining_after = bins - item
    
    # Only consider bins with enough remaining capacity
    valid_mask = remaining_after >= 0
    
    # If no valid bins, return all negative infinity
    if not np.any(valid_mask):
        return np.full(len(bins), -np.inf)
    
    n = len(bins)
    
    # Precompute sum and sum of squares of current remaining capacities
    sum_r = np.sum(bins)
    sum_sq_r = np.sum(bins * bins)
    
    # For each valid bin i, after placing the item:
    #   new_remaining_i = bins[i] - item
    #   new_sum = sum_r - item
    #   new_sum_sq = sum_sq_r - bins[i]^2 + (bins[i] - item)^2
    #              = sum_sq_r - 2*bins[i]*item + item^2
    #   new_mean = new_sum / n
    #   new_var = new_sum_sq / n - new_mean^2
    #
    # Since new_sum is the same for all i, new_mean is constant.
    # new_var = (sum_sq_r - 2*bins[i]*item + item^2) / n - (new_sum / n)^2
    #
    # To minimize new_var, we minimize (sum_sq_r - 2*bins[i]*item + item^2) / n
    # which is equivalent to minimizing -2*bins[i]*item + item^2
    # which is equivalent to maximizing bins[i] (for positive item).
    #
    # So the score is proportional to bins[i] (remaining capacity before placement).
    # We return bins[i] as the score for valid bins (higher = preferred).
    
    # Score: use the remaining capacity before placement as the score
    # This directly minimizes the resulting variance of the distribution
    scores = np.full(len(bins), -np.inf)
    scores[valid_mask] = bins[valid_mask]
    
    return scores
# EVOLVE-BLOCK-END
