"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items using a multi-objective dominance filter.
    
    Args:
        prizes: (5, 100) array of prize values, uniform in [0, 1]
        weights: (5, 100) array of weight values, uniform in [0, 1]
    
    Returns:
        (100,) array of attractiveness scores, one per item.
    """
    n_constraints, n_items = prizes.shape
    
    # Compute efficiency for each constraint: prize / weight
    # Avoid division by zero
    safe_weights = np.where(weights < 1e-9, 1e-9, weights)
    efficiencies = prizes / safe_weights  # shape (5, 100)
    
    # For each item, find the minimum efficiency across constraints (bottleneck)
    # This represents how "constrained" the item is by its worst-performing resource
    min_efficiency = efficiencies.min(axis=0)  # shape (100,)
    
    # Also compute the mean efficiency as a secondary metric
    mean_efficiency = efficiencies.mean(axis=0)  # shape (100,)
    
    # Identify Pareto-optimal items: an item is dominated if there exists another item
    # that has >= efficiency in all constraints and > in at least one
    # We'll use a simplified dominance check based on min and mean efficiency
    
    # For dominance, we need to check if item i is dominated by item j:
    # j has min_efficiency >= i's min_efficiency AND mean_efficiency >= i's mean_efficiency
    # AND (at least one is strictly greater)
    
    is_pareto = np.ones(n_items, dtype=bool)
    
    # For each item, check if it's dominated by any other item
    for i in range(n_items):
        if not is_pareto[i]:
            continue
        for j in range(n_items):
            if i == j or not is_pareto[j]:
                continue
            # Check if j dominates i
            if (min_efficiency[j] >= min_efficiency[i] and 
                mean_efficiency[j] >= mean_efficiency[i] and
                (min_efficiency[j] > min_efficiency[i] or mean_efficiency[j] > mean_efficiency[i])):
                is_pareto[i] = False
                break
    
    # Assign scores: Pareto-optimal items get high scores, others get lower
    # Use a combination of min_efficiency and mean_efficiency for ranking among Pareto items
    
    # Normalize efficiencies for scoring
    min_eff_norm = (min_efficiency - min_efficiency.min()) / (min_efficiency.max() - min_efficiency.min() + 1e-9)
    mean_eff_norm = (mean_efficiency - mean_efficiency.min()) / (mean_efficiency.max() - mean_efficiency.min() + 1e-9)
    
    # Combined score
    combined = 0.5 * min_eff_norm + 0.5 * mean_eff_norm
    
    # Base score for all items
    scores = 0.1 * combined
    
    # Boost Pareto-optimal items significantly
    scores[is_pareto] += 1.0 + combined[is_pareto]
    
    # Ensure all scores are positive
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
