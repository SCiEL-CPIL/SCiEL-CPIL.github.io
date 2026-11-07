"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    Combines two robust efficiency metrics:
    1. Aggregate Efficiency: Prize per total weight (Inspiration 1).
       Captures overall resource usage.
    2. Bottleneck Efficiency: Geometric mean of per-constraint prize/weight ratios (Inspiration 2).
       Captures the constraint that limits the item the most (smooth min).
    
    Both metrics are normalized to have a mean of 1 across items to ensure
    they contribute equally to the final heuristic score, preventing scale
    dominance by either metric.
    """
    eps = 1e-9
    
    # 1. Aggregate Efficiency: prizes / sum(weights)
    # Shape: (100,)
    total_weights = weights.sum(axis=0)
    sum_ratio = prizes / (total_weights + eps)
    
    # 2. Bottleneck Efficiency: Geometric Mean of (prizes / weights)
    # Shape: (5, 100)
    ratios = prizes / (weights + eps)
    
    # Geometric mean in log-space for stability
    log_ratios = np.log(ratios)
    log_geo_mean = log_ratios.mean(axis=0)
    geo_ratio = np.exp(log_geo_mean)
    
    # Normalize both to mean 1 to combine them fairly
    # Avoid division by zero if all values are identical (unlikely with random data)
    sum_ratio_mean = sum_ratio.mean()
    geo_ratio_mean = geo_ratio.mean()
    
    if sum_ratio_mean > 0:
        sum_ratio_norm = sum_ratio / sum_ratio_mean
    else:
        sum_ratio_norm = np.ones_like(sum_ratio)
        
    if geo_ratio_mean > 0:
        geo_ratio_norm = geo_ratio / geo_ratio_mean
    else:
        geo_ratio_norm = np.ones_like(geo_ratio)
        
    # Blend the two normalized metrics
    # Equal weight initially. This balances "total resource" vs "bottleneck resource".
    scores = 0.5 * sum_ratio_norm + 0.5 * geo_ratio_norm
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
