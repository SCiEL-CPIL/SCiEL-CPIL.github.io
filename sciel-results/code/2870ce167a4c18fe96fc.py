"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Args:
        prizes: Array of shape (C, N) or (N,), where C is constraints and N is items.
        weights: Array of shape (C, N).
        
    Returns:
        Array of shape (N,) with heuristic scores.
    """
    # Ensure prizes is treated as having the same number of rows as weights if needed for broadcasting
    # If prizes is 1D (N,), it broadcasts against weights (C, N).
    # If prizes is 2D (C, N), it matches weights (C, N).
    
    # Calculate value per item. If prizes is 2D, sum across constraints to get a total scalar value per item.
    # If prizes is 1D, it is already the scalar value.
    if prizes.ndim == 1:
        values = prizes
    else:
        values = prizes.sum(axis=0)
        
    # Add epsilon to weights to avoid division by zero
    eps = 1e-9
    safe_weights = weights + eps
    
    # Calculate ratio of value to weight for each constraint
    # ratios has shape (C, N)
    ratios = values / safe_weights
    
    # Use soft-min via log-sum-exp for a smooth bottleneck penalty
    # This creates a smoother landscape for the ACO probability distribution
    # Temperature parameter controls smoothness: lower = closer to hard min
    temperature = 0.1
    
    # Soft-min: -T * log(sum(exp(-x/T)))
    # For numerical stability, subtract max before exp
    neg_ratios = -ratios
    max_neg = np.max(neg_ratios, axis=0, keepdims=True)
    exp_neg = np.exp((neg_ratios - max_neg) / temperature)
    sum_exp = np.sum(exp_neg, axis=0)
    
    # Soft-min is -T * (log(sum_exp) + max_neg/T) = -T * log(sum_exp) - max_neg
    soft_min_scores = -temperature * np.log(sum_exp) - max_neg.squeeze(axis=0)
    
    # Ensure scores are positive for the ACO
    # Replace any non-finite or non-positive values with a small number
    scores = np.where(np.isfinite(soft_min_scores) & (soft_min_scores > 0), soft_min_scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
