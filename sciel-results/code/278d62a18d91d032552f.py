"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item using a multi-objective dominance filter.
    
    Approach:
    1. Compute total prize for each item (sum over constraints/resources).
    2. For each item, determine its "cost profile" across the 5 constraints.
    3. Identify Pareto-optimal items: an item is Pareto-optimal if no other item
       has >= prize in every constraint dimension AND strictly greater in at least one,
       while having <= weight in every constraint dimension AND strictly less in at least one.
       Actually, since we have 5 constraints and 100 items, we define dominance as:
       Item i dominates item j if for all constraints c, weight_i[c] <= weight_j[c]
       AND for at least one constraint c', weight_i[c'] < weight_j[c'],
       AND total_prize_i >= total_prize_j.
       But this is complex. Let's use a simpler but effective heuristic:
       
    Revised approach based on the directive:
    - Identify items that are not strictly dominated in prize-weight space.
    - For each item, compute a dominance score: how many other items it dominates.
    - Items with high dominance scores get high attractiveness.
    - Additionally, weight by prize-to-total-weight ratio to break ties.
    
    Implementation:
    1. total_prize[i] = sum of prizes across all 5 constraints for item i.
    2. total_weight[i] = sum of weights across all 5 constraints for item i.
    3. For each pair (i, j), check if i dominates j:
       - i has >= total_prize than j
       - i has <= total_weight than j
       - and at least one strict inequality
    4. dominance_count[i] = number of items j that i dominates.
    5. score[i] = (1 + dominance_count[i]) * (total_prize[i] / (total_weight[i] + eps))
       This gives high scores to items that dominate many others and have good ratio.
    6. Normalize and ensure positivity.
    """
    n_constraints, n_items = prizes.shape
    
    # Total prize and weight per item
    total_prize = prizes.sum(axis=0)  # shape (n_items,)
    total_weight = weights.sum(axis=0)  # shape (n_items,)
    
    eps = 1e-9
    
    # Base ratio: prize per unit total weight
    base_ratio = total_prize / (total_weight + eps)
    
    # Compute dominance counts
    # i dominates j if:
    #   total_prize[i] >= total_prize[j] AND total_weight[i] <= total_weight[j]
    #   AND (total_prize[i] > total_prize[j] OR total_weight[i] < total_weight[j])
    
    dominance_counts = np.zeros(n_items, dtype=np.float64)
    
    for i in range(n_items):
        # Items that i could dominate: those with <= prize and >= weight
        # and at least one strict
        prize_geq = total_prize >= total_prize[i]  # candidates j where prize_j <= prize_i... 
        # Wait, let me re-think: i dominates j means i is at least as good.
        # So we want j such that:
        #   total_prize[j] <= total_prize[i]
        #   total_weight[j] >= total_weight[i]
        #   AND (total_prize[j] < total_prize[i] OR total_weight[j] > total_weight[i])
        
        prize_leq = total_prize <= total_prize[i]
        weight_geq = total_weight >= total_weight[i]
        
        # Strict in at least one
        prize_strict = total_prize < total_prize[i]
        weight_strict = total_weight > total_weight[i]
        
        dominated = (prize_leq & weight_geq) & (prize_strict | weight_strict)
        # Exclude self
        dominated[i] = False
        dominance_counts[i] = np.sum(dominated)
    
    # Combine: high dominance count and good ratio
    # Use a combined score that emphasizes Pareto frontier items
    # Items with high dominance counts are on or near the Pareto frontier
    
    # Normalize dominance counts to [0, 1]
    max_dom = dominance_counts.max()
    if max_dom > 0:
        norm_dom = dominance_counts / max_dom
    else:
        norm_dom = np.ones(n_items)
    
    # Normalize base ratio to [0, 1]
    max_ratio = base_ratio.max()
    if max_ratio > 0:
        norm_ratio = base_ratio / max_ratio
    else:
        norm_ratio = np.ones(n_items)
    
    # Combined score: emphasize items that are both high ratio and high dominance
    # Use product to favor items that excel in both
    score = (1.0 + norm_dom) * (1.0 + norm_ratio)
    
    # Ensure all scores are positive
    score = np.maximum(score, 1e-9)
    
    return score
# EVOLVE-BLOCK-END
