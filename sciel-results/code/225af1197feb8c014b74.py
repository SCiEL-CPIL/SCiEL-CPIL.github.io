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
    2. Soft-min ratio: Smooth approximation of the minimum per-constraint ratio
       via log-sum-exp with a temperature parameter (captures bottleneck).
    
    The soft-min is weighted more heavily since the binding constraint
    dominates packing feasibility. Both are normalized via rank to be
    robust to outliers, then blended.
    """
    eps = 1e-9
    
    # Calculate ratios for each constraint: (5, 100)
    ratios = prizes / (weights + eps)
    
    # Metric 1: Sum-ratio (prize / total weight)
    total_weights = weights.sum(axis=0)
    sum_ratio = prizes / (total_weights + eps)
    
    # Metric 2: Soft-min of ratios via log-sum-exp (temperature-controlled)
    # soft_min(x, t) = -t * logsumexp(-x/t)
    # Lower t -> closer to true min (more bottleneck-focused)
    # t=1.0 gives behavior close to geometric mean; t=0.3 is more aggressive
    temperature = 0.5
    neg_ratios = -ratios / temperature
    # logsumexp along axis 0 (across 5 constraints)
    max_neg = neg_ratios.max(axis=0, keepdims=True)
    log_sum_exp = max_neg + np.log(np.exp(neg_ratios - max_neg).sum(axis=0))
    soft_min_ratio = -temperature * log_sum_exp.squeeze()
    
    # Rank-based normalization (robust to outliers)
    def rank_normalize(arr):
        """Normalize to [0, 1] via rank, then scale to mean 1."""
        ranks = np.argsort(np.argsort(arr)).astype(float)
        # ranks are 0..n-1; shift to 1..n for positive values
        ranks = ranks + 1.0
        # Normalize to mean 1
        return ranks / (ranks.mean() + eps)
    
    sum_ratio_norm = rank_normalize(sum_ratio)
    soft_min_norm = rank_normalize(soft_min_ratio)
    
    # Blend: weight the bottleneck (soft-min) more heavily
    # In multi-constraint knapsack, the most constrained dimension dominates
    w_bottleneck = 0.65
    w_sum = 1.0 - w_bottleneck
    scores = w_sum * sum_ratio_norm + w_bottleneck * soft_min_norm
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
