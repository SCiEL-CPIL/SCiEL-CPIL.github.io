"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Args:
        prizes: array of shape (5, 100) - prize values for each constraint/item
        weights: array of shape (5, 100) - weight values for each constraint/item
    
    Returns:
        array of shape (100,) - one attractiveness score per item
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure positive values to avoid division by zero
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    # Per-constraint efficiency: prize/weight for each constraint
    ratios = prizes / weights  # shape (5, 100)
    
    # Log of ratios for smooth operations
    log_ratios = np.log(ratios)
    
    # Soft-min across constraints (smooth surrogate for min-ratio)
    # We want to penalize items that are poor on any single constraint
    # softmin(x) = -logsumexp(-x)
    neg_log_ratios = -log_ratios
    max_neg = np.max(neg_log_ratios, axis=0)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_log_ratios - max_neg), axis=0))
    softmin_log = -log_sum_exp
    
    # Also compute the actual min ratio for reference
    min_ratios = np.min(ratios, axis=0)
    
    # Total weight per item (sum across constraints)
    total_weights = np.sum(weights, axis=0)
    
    # Total prize per item
    total_prizes = np.sum(prizes, axis=0)
    
    # Value density: total prize / total weight
    value_density = total_prizes / (total_weights + 1e-9)
    
    # Slack feasibility: how well an item fits in the tightest constraint
    # Capacity is 25 for each constraint, so slack = 25 - weight
    # We want items with small weights in the constraints where they have high weight
    # Slack ratio: (capacity - weight) / capacity for each constraint
    capacity = 25.0
    slack = (capacity - weights) / capacity  # shape (5, 100)
    
    # Minimum slack across constraints (bottleneck slack)
    min_slack = np.min(slack, axis=0)
    
    # Soft-min of slack for smoothness
    log_slack = np.log(np.maximum(slack, 1e-9))
    neg_log_slack = -log_slack
    max_neg_slack = np.max(neg_log_slack, axis=0)
    log_sum_exp_slack = max_neg_slack + np.log(np.sum(np.exp(neg_log_slack - max_neg_slack), axis=0))
    softmin_slack = -log_sum_exp_slack
    
    # Combine components:
    # 1. Value density (high is good)
    # 2. Soft-min ratio (high is good - item is efficient across all constraints)
    # 3. Min slack / soft-min slack (high is good - item fits well)
    
    # Normalize components to [0, 1] range for combination
    def normalize(x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min < 1e-12:
            return np.ones_like(x)
        return (x - x_min) / (x_max - x_min)
    
    # Weighted combination with emphasis on bottleneck feasibility
    score = (
        0.3 * normalize(value_density) +
        0.4 * normalize(np.exp(softmin_log)) +  # exp to bring back to ratio scale
        0.3 * normalize(np.exp(softmin_slack))  # slack feasibility
    )
    
    # Ensure all scores are positive
    score = np.maximum(score, 1e-9)
    
    return score
# EVOLVE-BLOCK-END
