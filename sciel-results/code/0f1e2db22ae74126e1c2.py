"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multi-constraint knapsack problem.
    
    The heuristic combines:
    1. Prize density (total prize / total weight)
    2. Penalty for bottleneck constraints (items heavy in any single constraint)
    3. Balance across constraints
    """
    # Total prize per item
    total_prize = prizes.sum(axis=0)
    
    # Total weight per item across all constraints
    total_weight = weights.sum(axis=0)
    
    # Max weight per item (bottleneck constraint)
    max_weight = weights.max(axis=0)
    
    # Mean weight per item
    mean_weight = weights.mean(axis=0)
    
    # Variance of weights across constraints (lower is more balanced)
    weight_var = np.var(weights, axis=0)
    
    # Base density: prize per unit total weight
    density = total_prize / (total_weight + 1e-9)
    
    # Bottleneck penalty: items that are very heavy in one constraint are penalized
    # Normalize max_weight by capacity (25) to get relative cost
    bottleneck_ratio = max_weight / 25.0
    bottleneck_penalty = 1.0 / (1.0 + bottleneck_ratio)
    
    # Balance bonus: prefer items with balanced weights across constraints
    # Lower variance means more balanced, so we reward low variance
    balance_score = 1.0 / (1.0 + weight_var)
    
    # Combine all factors
    # Weight the components: density is most important, then bottleneck, then balance
    heuristic = density * bottleneck_penalty * balance_score
    
    # Ensure all values are positive and finite
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
