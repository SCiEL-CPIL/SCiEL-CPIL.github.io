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
    
    Both are normalized and blended to balance scale and signal.
    """
    eps = 1e-9
    
    # Calculate ratios for each constraint: (5, 100)
    ratios = prizes / (weights + eps)
    
    # Metric 1: Sum-ratio (prize / total weight)
    total_weights = weights.sum(axis=0)
    sum_ratio = prizes / (total_weights + eps)
    
    # Metric 2: Geo-ratio (geometric mean of ratios)
    # This is a smooth approximation of min-ratio
    log_ratios = np.log(ratios + eps)
    geo_ratio = np.exp(log_ratios.mean(axis=0))
    
    # Normalize both to mean 1
    sum_ratio_norm = sum_ratio / (sum_ratio.mean() + eps)
    geo_ratio_norm = geo_ratio / (geo_ratio.mean() + eps)
    
    # Blend with equal weight
    scores = 0.5 * sum_ratio_norm + 0.5 * geo_ratio_norm
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
