"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    Combines three efficiency metrics with a dynamic scarcity adjustment:
    1. Aggregate Efficiency: Prize per total weight.
    2. Geometric Mean Efficiency: Smooth bottleneck surrogate (geometric mean of per-constraint ratios).
    3. Scarcity-Adjusted Score: Rewards items that are efficient in constraints that are
       globally scarce (i.e., where the top items are most efficient).
    
    The scarcity adjustment estimates the opportunity cost by looking at the distribution
    of efficiency across items for each constraint. If a constraint has a wide spread of
    efficiencies (high variance or high max-min gap), items that are efficient in that
    constraint are more valuable because they relieve a tighter bottleneck.
    """
    eps = 1e-9
    
    # 1. Aggregate Efficiency: prizes / sum(weights)
    total_weights = weights.sum(axis=0)
    sum_ratio = prizes / (total_weights + eps)
    
    # 2. Geometric Mean Efficiency: Geometric mean of (prizes / weights) per constraint
    ratios = prizes / (weights + eps)
    log_ratios = np.log(ratios)
    log_geo_mean = log_ratios.mean(axis=0)
    geo_ratio = np.exp(log_geo_mean)
    
    # 3. Scarcity-Adjusted Score
    # For each constraint, compute the "scarcity" as the spread of efficiencies.
    # A high spread means some items are much more efficient than others, making
    # the constraint more binding for less-efficient items.
    # We use the coefficient of variation (CV) or similar metric, but a simpler
    # proxy is the ratio of max to mean, or just the standard deviation.
    # Let's use the standard deviation of the ratios for each constraint.
    # If a constraint has high variance in ratios, it is "scarce" in the sense that
    # picking the wrong item hurts more.
    
    # Compute per-constraint variance of ratios
    # ratios shape: (5, 100)
    constraint_var = np.var(ratios, axis=1)  # shape: (5,)
    
    # Normalize variances to get a scarcity weight for each constraint
    # Higher variance -> higher scarcity weight
    # We want to boost items that are efficient in high-variance constraints.
    
    # Compute a "scarcity-weighted" ratio for each item
    # For each item, take the weighted sum of its per-constraint ratios,
    # where weights are the normalized variances.
    
    # Normalize variances to sum to 1 (or mean 1)
    var_sum = constraint_var.sum()
    if var_sum > 0:
        scarcity_weights = constraint_var / var_sum
    else:
        scarcity_weights = np.ones(5) / 5.0
    
    # Weighted sum of per-constraint ratios using scarcity weights
    # ratios: (5, 100), scarcity_weights: (5,)
    scarcity_adjusted = np.sum(ratios * scarcity_weights[:, np.newaxis], axis=0)
    
    # Normalize all three metrics to mean 1
    sum_ratio_mean = sum_ratio.mean()
    geo_ratio_mean = geo_ratio.mean()
    scarcity_mean = scarcity_adjusted.mean()
    
    if sum_ratio_mean > 0:
        sum_ratio_norm = sum_ratio / sum_ratio_mean
    else:
        sum_ratio_norm = np.ones_like(sum_ratio)
        
    if geo_ratio_mean > 0:
        geo_ratio_norm = geo_ratio / geo_ratio_mean
    else:
        geo_ratio_norm = np.ones_like(geo_ratio)
        
    if scarcity_mean > 0:
        scarcity_norm = scarcity_adjusted / scarcity_mean
    else:
        scarcity_norm = np.ones_like(scarcity_adjusted)
        
    # Blend the three normalized metrics
    # The geometric mean is structurally important for multi-constraint.
    # The scarcity adjustment adds dynamic information.
    # The aggregate efficiency is a simple baseline.
    scores = 0.3 * sum_ratio_norm + 0.4 * geo_ratio_norm + 0.3 * scarcity_norm
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
