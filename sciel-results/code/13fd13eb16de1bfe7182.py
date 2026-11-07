import numpy as np

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack problem.
    
    Args:
        prizes: array of shape (5, 100), values in [0, 1]
        weights: array of shape (5, 100), values in [0, 1]
    
    Returns:
        array of shape (100,), one attractiveness score per item (larger = more attractive)
    """
    capacity = 25.0
    n_items = weights.shape[1]
    
    # Total prize per item (sum across constraints)
    total_prize = np.sum(prizes, axis=0)
    
    # Relative weight per item and constraint
    rel_weights = weights / capacity
    
    # Bottleneck: max relative weight across constraints for each item
    bottleneck = np.max(rel_weights, axis=0)
    
    # How many constraints are "tight" for this item (close to max relative weight)
    # This captures the "multi-constraint burden" - items that stress multiple constraints
    # are harder to fit with others
    threshold = bottleneck * 0.9  # constraints within 90% of the max are considered tight
    tight_count = np.sum(rel_weights >= threshold, axis=0)
    
    # Efficiency: prize per unit of bottleneck weight
    epsilon = 1e-9
    efficiency = total_prize / (bottleneck + epsilon)
    
    # Penalty for multi-constraint burden:
    # Items that are tight in many constraints reduce flexibility for future items
    # Scale penalty by how many constraints are tight
    flexibility_penalty = tight_count / 5.0  # 0 to 1
    
    # Combine: high efficiency, low flexibility penalty
    # Use a multiplicative form to balance the two factors
    heuristic = efficiency * (1.0 - 0.5 * flexibility_penalty)
    
    # Also consider: items with low total relative weight across all constraints
    # are "light" and leave more room for other items
    total_rel_weight = np.sum(rel_weights, axis=0)
    lightness_bonus = 1.0 / (total_rel_weight + epsilon)
    
    # Final heuristic: combine efficiency, flexibility, and lightness
    # Normalize components to comparable scales
    eff_norm = efficiency / (np.max(efficiency) + epsilon)
    light_norm = lightness_bonus / (np.max(lightness_bonus) + epsilon)
    flex_score = 1.0 - flexibility_penalty
    
    # Weighted combination
    heuristic = 0.5 * eff_norm + 0.3 * light_norm + 0.2 * flex_score
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, epsilon)
    
    return heuristic
