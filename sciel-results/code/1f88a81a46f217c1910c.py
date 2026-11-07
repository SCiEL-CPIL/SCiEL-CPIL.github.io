"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Each constraint has capacity 25, 100 items
    
    # Total prize per item (sum across constraints, but actually prizes[i,j] is the prize
    # contribution of item j to constraint i? No - prizes and weights are both (5,100).
    # The task says "prizes and weights, of shape (5, 100), are uniform in [0, 1]"
    # and "every constraint has capacity 25"
    # 
    # In a multi-constraint knapsack: each item has a weight in each constraint dimension
    # and a single prize value. But here prizes is (5,100) too.
    # 
    # Looking at the ACO: "every ant adds items that still fit in all constraints"
    # "best prize of that iteration" - so each item has a total prize.
    # 
    # The prizes (5,100) likely means: prize[i,j] is the prize contribution of item j
    # in dimension i? Or more likely, the total prize of item j is sum(prizes[:,j])
    # or maybe prizes[i,j] is the prize for item j and it's repeated? 
    #
    # Actually re-reading: "mean over instances of the highest total prize found by any ant"
    # The total prize of a solution = sum of prizes of selected items.
    # If prizes is (5,100), the total prize is probably sum(prizes, axis=0) per item,
    # or prizes might represent per-constraint prize contributions.
    #
    # Given the structure, I'll assume total_prize[j] = sum(prizes[:, j])
    # and weight in constraint i for item j = weights[i, j]
    
    total_prize = prizes.sum(axis=0)  # (100,)
    total_weight = weights.sum(axis=0)  # (100,)
    
    # Find the bottleneck constraint: the one with highest mean weight
    mean_weights_per_constraint = weights.mean(axis=1)  # (5,)
    bottleneck_idx = np.argmax(mean_weights_per_constraint)
    bottleneck_weights = weights[bottleneck_idx, :]  # (100,)
    
    # Efficiency on bottleneck
    eps = 1e-9
    bottleneck_eff = total_prize / (bottleneck_weights + eps)
    
    # Overall efficiency
    overall_eff = total_prize / (total_weight + eps)
    
    # Penalty for consuming disproportionate share of bottleneck capacity
    # Capacity is 25, so an item with bottleneck weight > 1 is relatively heavy
    # Normalize bottleneck weights: fraction of capacity
    bottleneck_fraction = bottleneck_weights / 25.0  # fraction of capacity used
    
    # Penalize items that use too much of the bottleneck
    # Use a smooth penalty: items with high bottleneck_fraction get reduced score
    # The penalty should be mild for small fractions, stronger for large ones
    penalty = 1.0 / (1.0 + 2.0 * bottleneck_fraction**2)
    
    # Combine: weight bottleneck efficiency more heavily since it's the binding constraint
    # Blend with overall efficiency
    combined = 0.7 * bottleneck_eff + 0.3 * overall_eff
    
    # Apply penalty
    score = combined * penalty
    
    # Ensure all values are positive and finite
    score = np.maximum(score, eps)
    score = np.where(np.isfinite(score), score, eps)
    
    # Normalize to have a reasonable scale (mean around 1)
    mean_score = score.mean()
    if mean_score > eps:
        score = score / mean_score
    
    return score
# EVOLVE-BLOCK-END
