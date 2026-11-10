import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes heuristic scores for items in a Multiple-Constraint Knapsack Problem.
    
    Args:
        prizes: np.array of shape (5, 100), uniform in [0, 1].
        weights: np.array of shape (5, 100), uniform in [0, 1].
        
    Returns:
        np.array of shape (100,), heuristic scores.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure positive weights to avoid division by zero
    weights_safe = np.maximum(weights, 1e-9)
    
    # 1. Total prize and weight per item
    total_prize = np.sum(prizes, axis=0)
    total_weight = np.sum(weights_safe, axis=0)
    
    # 2. Density: Average value per unit weight
    density = total_prize / (total_weight + 1e-9)
    
    # 3. Bottleneck Analysis
    # Ratio of prize to weight for each constraint
    ratios = prizes / weights_safe
    
    # The "bottleneck" constraint for an item is the one with the lowest ratio
    # Items with a very low min_ratio are inefficient in at least one dimension
    min_ratio = np.min(ratios, axis=0)
    
    # 4. Weight Slack
    # Penalize items that consume a large portion of any single constraint's capacity
    # Capacity is 25
    max_weight = np.max(weights_safe, axis=0)
    # Slack is 1 when weight is 0, approaches 0 when weight approaches 25
    # Since weights are in [0,1], max_weight is <= 1, so slack is in [0.96, 1]
    # To make the penalty more significant, we can scale it or use a different metric
    # However, the knowledge suggests this term helped.
    weight_slack = 1.0 - (max_weight / 25.0)
    
    # Ensure slack is positive
    weight_slack = np.maximum(weight_slack, 1e-9)
    
    # 5. Combine terms
    # Formula: prize * slack * density^1.5 * bottleneck_factor
    # We use min_ratio^0.5 as the bottleneck factor to penalize weak constraints
    bottleneck_factor = np.power(np.maximum(min_ratio, 1e-9), 0.5)
    
    scores = total_prize * weight_slack * np.power(density, 1.5) * bottleneck_factor
    
    # Ensure scores are finite and positive
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    scores = np.maximum(scores, 1e-9)
    
    return scores
