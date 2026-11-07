"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Args:
        prizes: numpy array of shape (5, 100) with prizes for each constraint/item.
        weights: numpy array of shape (5, 100) with weights for each constraint/item.
        
    Returns:
        numpy array of shape (100,) with heuristic scores (larger is more attractive).
    """
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)  # (100,)
    
    # Cost metrics:
    # 1. Total weight across all constraints (volume)
    total_weight = weights.sum(axis=0)  # (100,)
    # 2. Maximum weight in any single constraint (bottleneck/shape inefficiency)
    max_weight = weights.max(axis=0)  # (100,)
    
    # Combined cost: penalize both total consumption and peak constraint usage
    # The max_weight term is important because an item that heavily uses one constraint
    # limits future choices even if its total weight is moderate
    combined_cost = total_weight + 0.5 * max_weight
    
    # Heuristic: prize density with combined cost
    heuristic = total_prize / (combined_cost + 1e-9)
    
    # Ensure all values are positive (ACO requires positive heuristics)
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Normalize to have mean 1 for numerical stability with pheromone
    mean_h = np.mean(heuristic)
    if mean_h > 0:
        heuristic = heuristic / mean_h
        
    return heuristic
# EVOLVE-BLOCK-END
