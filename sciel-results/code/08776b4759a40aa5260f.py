"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ant colony optimization of
    multiple-constraint knapsack.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0,1]
        weights: array of shape (5, 100), uniform in [0,1]
    
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure positive values
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    # Per-constraint efficiency: prize/weight for each constraint
    # Shape: (5, 100)
    ratios = prizes / weights
    
    # Log of ratios for smooth operations
    log_ratios = np.log(ratios)
    
    # Soft-min across constraints using log-sum-exp trick
    # softmin(x) = -logsumexp(-x)
    # We want to penalize items that are poor on any single constraint
    neg_log_ratios = -log_ratios
    
    # Temperature for soft-min: lower = sharper
    temperature = 0.05
    
    # Compute soft-min: -logsumexp(-log_ratios / temperature) * temperature
    # = -logsumexp(neg_log_ratios / temperature) * temperature
    scaled = neg_log_ratios / temperature
    
    # Numerically stable logsumexp
    max_scaled = np.max(scaled, axis=0, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(scaled - max_scaled), axis=0)) + max_scaled
    
    # Soft-min value
    soft_min = -logsumexp[0] * temperature
    
    # Exponentiate to get back to ratio scale
    soft_min_ratio = np.exp(soft_min)
    
    # Bottleneck penalty: square the soft-min ratio to emphasize bottleneck constraints
    bottleneck_penalty = soft_min_ratio ** 2
    
    # Combine with total prize as a base attractiveness
    total_prize = np.sum(prizes, axis=0)
    
    # Final heuristic: emphasize items with high total prize AND good bottleneck
    # Use a sharpening power to differentiate items more
    heuristic = (total_prize ** 3) * bottleneck_penalty
    
    # Ensure all values are positive and finite
    heuristic = np.maximum(heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
