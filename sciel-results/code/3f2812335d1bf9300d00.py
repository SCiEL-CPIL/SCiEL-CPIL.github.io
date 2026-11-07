"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack.
    
    Uses a Lagrangian relaxation approach where the heuristic is the total prize
    divided by the sum of normalized weights across all constraints.
    Normalization ensures that constraints with larger average weights are 
    appropriately scaled relative to those with smaller average weights.
    
    Args:
        prizes: (5, 100) array of prize values per constraint/item
        weights: (5, 100) array of weight values per constraint/item
    
    Returns:
        (100,) array of heuristic scores, higher = more attractive
    """
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)
    
    # Normalize weights per constraint to account for different scales
    # Mean weight per constraint
    mean_weights = np.mean(weights, axis=1, keepdims=True) # Shape (5, 1)
    
    # Avoid division by zero
    mean_weights = np.where(mean_weights < 1e-9, 1e-9, mean_weights)
    
    # Normalized weights: weights[c, i] / mean_weights[c]
    # This makes the average normalized weight 1 for each constraint
    norm_weights = weights / mean_weights # Shape (5, 100)
    
    # Sum of normalized weights represents the "total cost" of the item
    # in a space where all constraints are equally scaled
    total_norm_weight = norm_weights.sum(axis=0) # Shape (100,)
    
    # Heuristic: Prize / Total Normalized Weight
    # We add a small epsilon to denominator to avoid division by zero
    # and to handle items with zero weight (though unlikely with uniform [0,1])
    eps = 1e-9
    h = total_prize / (total_norm_weight + eps)
    
    # Ensure all values are positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
