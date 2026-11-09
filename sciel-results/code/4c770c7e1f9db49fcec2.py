"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores using Pareto dominance filtering.
    
    Approach:
    1. Identify the most constraining resource for each item (binding constraint).
    2. Compute prize-per-unit-weight for the binding constraint.
    3. Apply Pareto dominance filter: items not dominated by others get high scores.
    4. Scale scores by efficiency within the Pareto front.
    """
    n_constraints, n_items = prizes.shape
    capacity = 25.0
    
    # Step 1: Find the most constraining resource for each item
    weight_fraction = weights / capacity
    binding_idx = np.argmax(weight_fraction, axis=0)
    
    # Step 2: Compute prize per unit weight for the binding constraint
    binding_prize = np.array([prizes[binding_idx[i], i] for i in range(n_items)])
    binding_weight = np.array([weights[binding_idx[i], i] for i in range(n_items)])
    efficiency = binding_prize / (binding_weight + 1e-9)
    
    # Step 3: Pareto dominance filter (O(n^2), fine for n=100)
    is_dominated = np.zeros(n_items, dtype=bool)
    
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            # Check if j dominates i
            prize_geq = np.all(prizes[:, j] >= prizes[:, i] - 1e-12)
            weight_leq = np.all(weights[:, j] <= weights[:, i] + 1e-12)
            prize_strict = np.any(prizes[:, j] > prizes[:, i] + 1e-12)
            weight_strict = np.any(weights[:, j] < weights[:, i] - 1e-12)
            
            if prize_geq and weight_leq and (prize_strict or weight_strict):
                is_dominated[i] = True
                break
    
    # Step 4: Assign scores
    scores = np.ones(n_items, dtype=float)
    
    nondom_mask = ~is_dominated
    dom_mask = is_dominated
    
    if np.any(nondom_mask):
        nondom_eff = efficiency[nondom_mask]
        max_eff = np.max(nondom_eff) if np.max(nondom_eff) > 0 else 1.0
        scores[nondom_mask] = 1.0 + 9.0 * (nondom_eff / (max_eff + 1e-9))
    
    if np.any(dom_mask):
        dom_eff = efficiency[dom_mask]
        max_eff_all = np.max(efficiency) if np.max(efficiency) > 0 else 1.0
        scores[dom_mask] = 0.1 + 0.4 * (dom_eff / (max_eff_all + 1e-9))
    
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
