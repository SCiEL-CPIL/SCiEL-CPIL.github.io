"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)
    
    # For each item, find the most constraining dimension
    # An item is limited by its maximum weight across all constraints
    max_weight = weights.max(axis=0)
    
    # Average weight per item (to account for overall resource usage)
    avg_weight = weights.mean(axis=0)
    
    # Heuristic: balance between prize density and constraint risk
    # Use max weight as primary penalty (bottleneck constraint)
    # But also consider average weight to avoid items that are heavy overall
    
    # Combine: reward high total prize, penalize both max and average weight
    # Use a weighted combination to balance between the two
    penalty = 0.7 * max_weight + 0.3 * avg_weight
    
    # Compute heuristic score
    h = total_prize / (penalty + 1e-9)
    
    # Ensure all values are positive (as required by task)
    h[h <= 0] = 1e-9
    
    return h
# EVOLVE-BLOCK-END
