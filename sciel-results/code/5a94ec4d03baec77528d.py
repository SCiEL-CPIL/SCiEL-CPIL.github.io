"""Rewritten program for ant colony optimization heuristic scoring."""
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
    # Ensure inputs are valid
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Input shapes must be (5, 100)")
    
    # Total prize per item (sum across all 5 constraints)
    total_prize = prizes.sum(axis=0)
    
    # For each item, compute the "cost" in terms of how much it consumes
    # of the tightest constraint. Since all capacities are 25, we can
    # normalize weights by capacity to get a fraction of capacity used.
    
    # Max weight across constraints for each item - this is the bottleneck
    max_weight = weights.max(axis=0)
    
    # Sum of weights across all constraints for each item
    sum_weight = weights.sum(axis=0)
    
    # Mean weight across constraints
    mean_weight = weights.mean(axis=0)
    
    # A good heuristic should favor items with high prize-to-weight ratio
    # But also consider the multi-dimensional nature: an item that uses a lot
    # in one dimension but little in others might be better than one that
    # uses moderately in all dimensions.
    
    # Strategy: Use a combination of:
    # 1. Prize density: total_prize / max_weight (how much prize per unit of 
    #    the most constraining resource)
    # 2. Prize density: total_prize / sum_weight (how much prize per total 
    #    resource consumption)
    # 3. A penalty for items that are "bottlenecked" - i.e., where max_weight
    #    is much larger than mean_weight (high variance in weight usage)
    
    # Compute prize densities
    density_max = total_prize / (max_weight + 1e-9)
    density_sum = total_prize / (sum_weight + 1e-9)
    
    # Variance of weights across constraints for each item
    weight_var = weights.var(axis=0)
    
    # Coefficient of variation for weights
    weight_cv = np.sqrt(weight_var) / (mean_weight + 1e-9)
    
    # Combine: higher density is better, lower CV is better (more balanced usage)
    # We want to penalize items that are heavily skewed toward one constraint
    
    # Normalize the densities to [0, 1] range for combining
    density_max_norm = (density_max - density_max.min()) / (density_max.max() - density_max.min() + 1e-9)
    density_sum_norm = (density_sum - density_sum.min()) / (density_sum.max() - density_sum.min() + 1e-9)
    
    # Normalize CV to [0, 1] (lower CV is better, so invert)
    cv_norm = (weight_cv - weight_cv.min()) / (weight_cv.max() - weight_cv.min() + 1e-9)
    cv_inverted = 1.0 - cv_norm
    
    # Final heuristic: weighted combination
    # Heavier weight on density_max since the bottleneck constraint is most important
    heuristic = (
        0.5 * density_max_norm +
        0.3 * density_sum_norm +
        0.2 * cv_inverted
    )
    
    # Ensure all values are positive (as required by the task)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
