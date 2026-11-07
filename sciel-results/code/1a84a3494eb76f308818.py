"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items using a Greedy Simulation strategy.
    
    Strategy:
    1. Calculate a static "efficiency" score (prize sum / weight sum) to guide a greedy simulation.
    2. Simulate a greedy packing of items based on this efficiency, respecting all 5 constraints (capacity 25).
    3. The heuristic score for each item is derived from its role in this greedy solution:
       - Items selected in the greedy solution get a high base score, scaled by their marginal prize contribution.
       - Items not selected get a lower score, proportional to their potential "missed opportunity" or simply low.
    4. This dynamic, simulation-based score better reflects the combinatorial reality than a static ratio.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 300s limit
    
    n_constraints, n_items = prizes.shape
    capacities = np.ones(n_constraints) * 25.0
    
    # 1. Static Efficiency Score for Greedy Simulation
    # Total prize per item
    total_prizes = np.sum(prizes, axis=0)
    # Total weight per item (sum across constraints)
    total_weights = np.sum(weights, axis=0)
    
    # Avoid division by zero
    total_weights_safe = np.where(total_weights > 1e-9, total_weights, 1.0)
    
    # Efficiency: prize per unit total weight
    efficiency = total_prizes / total_weights_safe
    
    # 2. Greedy Simulation
    # Sort items by efficiency in descending order
    sorted_indices = np.argsort(-efficiency)
    
    remaining_capacities = capacities.copy()
    selected_items = np.zeros(n_items, dtype=bool)
    greedy_prize = 0.0
    
    for idx in sorted_indices:
        # Check if item fits in all constraints
        if np.all(weights[:, idx] <= remaining_capacities):
            remaining_capacities -= weights[:, idx]
            selected_items[idx] = True
            greedy_prize += total_prizes[idx]
            
            # Optional: Early stop if we've filled most capacity or time is running out
            if time.time() > deadline:
                break
    
    # 3. Construct Heuristic Scores
    # We want to guide ants towards the "core" items of the greedy solution.
    # However, a pure binary mask (1 for selected, 0 for not) is too coarse and might
    # cause all ants to pick the exact same set, reducing exploration.
    
    # Strategy: 
    # - Selected items get a score proportional to their prize/weight ratio (efficiency) 
    #   but boosted to ensure they are strongly preferred.
    # - Unselected items get a score based on their efficiency but damped, 
    #   allowing ants to consider them if pheromone is high, but generally preferring selected.
    
    # Normalize efficiency to [0, 1] range for scaling
    eff_min = np.min(efficiency)
    eff_max = np.max(efficiency)
    if eff_max > eff_min:
        norm_eff = (efficiency - eff_min) / (eff_max - eff_min)
    else:
        norm_eff = np.ones(n_items)
    
    # Base score: normalized efficiency
    scores = norm_eff
    
    # Boost selected items
    # The boost factor should be significant to steer ants towards the greedy core
    # but not so high that it ignores pheromone entirely.
    # Let's use a multiplier of 2.0 for selected items, relative to their normalized efficiency.
    # Actually, a better approach: 
    # Score = (Selected ? 1.0 : 0.1) * (1 + norm_eff)
    # This creates a clear hierarchy: Selected items are always much more attractive than unselected.
    
    scores[selected_items] = 1.0 + norm_eff[selected_items]
    scores[~selected_items] = 0.1 + 0.5 * norm_eff[~selected_items]
    
    # Ensure all values are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
