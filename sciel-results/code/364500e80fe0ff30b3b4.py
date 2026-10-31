"""Improved program using structural centrality (variance minimization) heuristic."""
# EVOLVE-BLOCK-START
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores bins based on structural centrality:
    We prefer placing the item in a bin such that its new remaining capacity
    is close to the mean remaining capacity of all other bins.
    This minimizes the variance of the remaining capacities, keeping the
    packing state "homogeneous" and potentially more robust for future items
    drawn from a Weibull distribution.
    """
    remaining = bins
    valid_mask = remaining >= item
    
    # If no bin can fit the item, return -inf for all
    if not np.any(valid_mask):
        return -np.inf * np.ones_like(remaining)
    
    n_bins = len(remaining)
    if n_bins == 0:
        return np.array([])
    
    # Calculate total sum and mean of all bins
    total_sum = np.sum(remaining)
    
    # Precompute scores for valid bins
    scores = np.full(n_bins, -np.inf)
    
    # Indices of valid bins
    valid_indices = np.where(valid_mask)[0]
    
    # If only one bin is valid, it must be chosen
    if len(valid_indices) == 1:
        scores[valid_indices[0]] = 0.0
        return scores

    # For each valid bin, calculate the "centrality" score
    # We want (R_i - item) to be close to mean(R_j for j != i)
    
    # Vectorized calculation for all bins first, then mask
    # R_i is remaining
    # new_remainder_i = remaining - item
    new_remainders = remaining - item
    
    # Sum of others for each bin i: total_sum - remaining[i]
    sum_others = total_sum - remaining
    
    # Count of others is n_bins - 1
    count_others = n_bins - 1
    
    # Mean of others for each bin i
    # Avoid division by zero if n_bins == 1, but we handled that above
    mean_others = sum_others / count_others
    
    # Deviation of new remainder from mean of others
    deviations = new_remainders - mean_others
    
    # Score is negative squared deviation
    # We only care about valid bins
    centrality_scores = - (deviations ** 2)
    
    # Apply to valid bins
    scores[valid_indices] = centrality_scores[valid_indices]
    
    # Refinement: Strongly prefer exact fits
    # If remaining == item, the deviation is (0 - mean_others). 
    # If mean_others is large, this score is very negative.
    # But an exact fit is always optimal locally.
    # We should override the score for exact fits to be very high.
    exact_fit_mask = valid_mask & (remaining == item)
    if np.any(exact_fit_mask):
        # Set exact fits to a very high score
        # We need to ensure this is higher than any other centrality score.
        # The max centrality score is 0 (if deviation is 0).
        # So any value > 0 is safe, but let's use a large constant.
        scores[exact_fit_mask] = 1e9
        
    # Refinement: If no exact fit, we might want to blend with Best Fit?
    # The directive says "structural change". Pure centrality is the change.
    # However, if the mean of others is very different from the best fit,
    # centrality might pick a "worse" fit.
    # Example: Bins [10, 100]. Item 10.
    # Bin 0: new=0. Others [100]. Mean=100. Dev= -100. Score=-10000.
    # Bin 1: new=90. Others [10]. Mean=10. Dev= 80. Score=-6400.
    # Best Fit picks Bin 0 (new=0).
    # Centrality picks Bin 1 (Score -6400 > -10000).
    # Is Bin 1 better?
    # State after Bin 0: [0, 100].
    # State after Bin 1: [10, 90].
    # If next item is 10, Bin 1 state allows filling Bin 0 (10) and Bin 1 (90->80).
    # Bin 0 state allows filling Bin 1 (100->90).
    # Both seem fine. But [10, 90] has lower variance than [0, 100].
    # Variance of [0, 100] is 2500. Variance of [10, 90] is 1600.
    # So Centrality prefers the lower variance state.
    # This is a valid structural heuristic.
    
    return scores
# EVOLVE-BLOCK-END
