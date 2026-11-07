"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    Combines two efficiency metrics:
    1. Sum-ratio: Prize per total weight (overall resource usage).
    2. Geo-ratio: Geometric mean of per-constraint prize/weight ratios 
       (smooth min, captures bottleneck constraint).
    
    The final score is a normalized blend of these two signals.
    """
    eps = 1e-9
    
    # Calculate ratios for each constraint: (5, 100)
    ratios = prizes / (weights + eps)
    
    # Sum-ratio: prize / total weight across all constraints
    sum_ratio = prizes.sum(axis=0) / (weights.sum(axis=0) + eps)
    
    # Geo-ratio: geometric mean of per-constraint ratios
    # log(geo_mean) = mean(log(ratio))
    log_ratios = np.log(ratios)
    geo_ratio = np.exp(log_ratios.mean(axis=0))
    
    # Normalize both to mean 1
    sum_ratio_mean = sum_ratio.mean()
    if sum_ratio_mean > 0 and np.isfinite(sum_ratio_mean):
        sum_ratio = sum_ratio / sum_ratio_mean
    else:
        sum_ratio = np.ones_like(sum_ratio)
        
    geo_ratio_mean = geo_ratio.mean()
    if geo_ratio_mean > 0 and np.isfinite(geo_ratio_mean):
        geo_ratio = geo_ratio / geo_ratio_mean
    else:
        geo_ratio = np.ones_like(geo_ratio)
    
    # Blend the two metrics with equal weight
    scores = 0.5 * sum_ratio + 0.5 * geo_ratio
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
