"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for each item in a multiple-constraint knapsack.
    
    Args:
        prizes: array of shape (5, 100) - prize values for each constraint/item
        weights: array of shape (5, 100) - weight costs for each constraint/item
    
    Returns:
        array of shape (100,) - attractiveness score per item (larger = more attractive)
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)
    
    # For each item, compute the maximum resource consumption across constraints
    # An item is "expensive" if it consumes a lot of any single resource
    max_weight = weights.max(axis=0)
    avg_weight = weights.mean(axis=0)
    
    # Efficiency: prize per unit of the most constraining resource
    # We want high prize and low weight on all dimensions
    # Use the minimum ratio across constraints as a robust efficiency measure
    # ratio_i = prize_i / weight_i for each constraint i
    # But weights can be near 0, so add small epsilon
    
    eps = 1e-9
    safe_weights = weights + eps
    ratios = prizes / safe_weights  # shape (5, 100)
    
    # The minimum ratio indicates the worst-case efficiency across constraints
    min_ratio = ratios.min(axis=0)
    
    # Also consider the geometric mean of ratios for a balanced view
    log_ratios = np.log(ratios)
    geo_mean_ratio = np.exp(log_ratios.mean(axis=0))
    
    # Combine: emphasize items that are efficient across ALL constraints
    # Use a product-like combination: min_ratio * geo_mean_ratio
    # This penalizes items that are bad on any single constraint
    
    score = min_ratio * geo_mean_ratio
    
    # Also factor in total prize to prefer high-value items
    # Normalize total_prize to [0, 1] range for scaling
    tp_min = total_prize.min()
    tp_max = total_prize.max()
    tp_range = tp_max - tp_min
    if tp_range > eps:
        norm_prize = (total_prize - tp_min) / tp_range
    else:
        norm_prize = np.ones_like(total_prize)
    
    # Final score: efficiency-dominant, with prize as a secondary factor
    # Weight efficiency heavily since it determines feasibility
    score = score * (1.0 + norm_prize)
    
    # Ensure all values are positive and finite
    score = np.where(np.isfinite(score) & (score > 0), score, 1e-9)
    
    return score
# EVOLVE-BLOCK-END
