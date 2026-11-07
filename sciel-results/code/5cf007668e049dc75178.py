"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    The heuristic is designed to guide ant colony optimization by scoring items
    based on their prize-to-weight ratio, with emphasis on the bottleneck constraint.
    
    Args:
        prizes: (5, 100) array of prize values in [0, 1]
        weights: (5, 100) array of weight values in [0, 1]
        
    Returns:
        (100,) array of attractiveness scores (larger = more attractive)
    """
    # Compute per-constraint prize-to-weight ratios
    # For each constraint, an item's "efficiency" is prize / weight
    # We want to identify items that are efficient across all constraints
    
    # Avoid division by zero
    weights_safe = np.maximum(weights, 1e-9)
    
    # Per-constraint efficiency: prize / weight for each constraint
    efficiency = prizes / weights_safe  # shape (5, 100)
    
    # The bottleneck constraint is the one that limits us most.
    # An item is "balanced" if it has good efficiency across all constraints.
    # An item is "specialized" if it's great in one constraint but poor in others.
    
    # Strategy: Use the minimum efficiency across constraints as a base score,
    # but also consider the average to reward balanced items.
    
    # min_efficiency: worst-case efficiency across constraints
    min_efficiency = efficiency.min(axis=0)  # shape (100,)
    
    # avg_efficiency: average efficiency across constraints  
    avg_efficiency = efficiency.mean(axis=0)  # shape (100,)
    
    # Combine: emphasize the bottleneck (min) but also reward overall quality (avg)
    # This makes balanced items more attractive than specialized ones
    score = min_efficiency + 0.5 * avg_efficiency
    
    # Ensure all scores are positive
    score = np.maximum(score, 1e-9)
    
    return score
# EVOLVE-BLOCK-END
