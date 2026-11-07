"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)
    
    # Capacity is 25 for each constraint
    capacity = 25.0
    
    # For each constraint, compute the sum of weights across all items
    # This tells us how "tight" each constraint is overall
    weight_sums = weights.sum(axis=1)  # shape (5,)
    
    # Residual capacity for each constraint
    residual = capacity - weight_sums  # shape (5,)
    
    # Clip residual to be at least a small positive value to avoid division issues
    residual = np.maximum(residual, 1e-6)
    
    # Per-constraint prize-to-weight ratio
    # For each constraint c and item i: prizes[c, i] / (weights[c, i] + eps)
    eps = 1e-9
    prize_to_weight = prizes / (weights + eps)  # shape (5, 100)
    
    # Scale by residual capacity: items that use resources with more available capacity
    # get a higher score
    # weighted_ratio[c, i] = prize_to_weight[c, i] * residual[c]
    weighted_ratio = prize_to_weight * residual[:, np.newaxis]  # shape (5, 100)
    
    # Sum across constraints to get a composite score
    # This rewards items that have good prize-to-weight ratios in constraints
    # where there is still plenty of capacity available
    h = weighted_ratio.sum(axis=0)  # shape (100,)
    
    # Ensure all values are positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
