"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores using a Greedy Forward Simulation.
    
    Strategy:
    1. The ACO algorithm constructs solutions sequentially. An item's attractiveness
       should reflect its marginal contribution to a feasible solution built in a 
       typical order.
    2. We simulate a "representative" greedy construction:
       - Calculate a base priority for each item (e.g., prize sum / weight sum).
       - Iterate through items in descending order of priority.
       - If an item fits the remaining capacities, "select" it and update remaining capacities.
       - Assign a high score to selected items, reflecting that they are part of a 
         high-quality greedy core.
       - Assign lower scores to items that were skipped due to constraints, but 
         modulated by their potential prize if they had been lighter.
    3. This creates a heuristic that distinguishes between items that are "core" 
       (likely to be picked early by any ant) and "filler" (only picked if space remains).
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Safety check for shape and finite values
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        # Fallback: simple ratio
        total_prizes = np.sum(prizes, axis=0)
        total_weights = np.sum(weights, axis=0)
        heuristic = total_prizes / (total_weights + 1e-9)
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        
    if not (np.all(np.isfinite(prizes)) and np.all(np.isfinite(weights))):
        return np.ones(100) * 1e-9

    # 1. Calculate Base Priority
    # We use the sum of prizes over the sum of weights as a density metric.
    # This is a standard MCKP heuristic.
    total_prizes_per_item = np.sum(prizes, axis=0)  # shape (100,)
    total_weights_per_item = np.sum(weights, axis=0)  # shape (100,)
    
    # Avoid division by zero
    safe_weights = np.where(total_weights_per_item > 1e-9, total_weights_per_item, 1e-9)
    base_priority = total_prizes_per_item / safe_weights
    
    # 2. Greedy Simulation
    # Sort items by base_priority in descending order
    order = np.argsort(-base_priority)
    
    # Initialize remaining capacities (5 constraints, capacity 25 each)
    capacity = 25.0
    remaining_cap = np.full(5, capacity, dtype=np.float64)
    
    # Initialize heuristic scores
    heuristic = np.zeros(100, dtype=np.float64)
    
    # We will assign scores based on whether the item is in the "greedy core"
    # and its position in the selection.
    
    # To make the heuristic more robust, we can run this simulation a few times 
    # with slightly perturbed priorities or just use one deterministic pass 
    # but weight the scores by "scarcity" encountered.
    
    # Let's stick to one deterministic pass for speed, but refine the scoring.
    # Items selected get a score proportional to their prize.
    # Items not selected get a score proportional to their prize * (probability they fit if constraints were looser).
    
    # A better approach: 
    # The heuristic should guide the ant. If an item is in the greedy optimal solution, 
    # it should have a high heuristic. If it's not, it should have a lower one.
    # However, the ACO might find a different combination. 
    # A common effective heuristic is the "regret" or "opportunity cost".
    
    # Let's try a different structural change: 
    # **Multi-Objective Ranking**
    # Instead of a single scalar density, we rank items by their performance in 
    # each constraint individually and combine them.
    
    # Alternative Strategy: **Constraint-Aware Ranking**
    # For each constraint j, calculate the "cost" of an item.
    # An item is good if it has high prize and low cost in the *most binding* constraints.
    
    # Let's identify the most binding constraints in the greedy solution.
    
    # Re-doing the simulation to track which constraints are tight.
    selected_items = []
    for idx in order:
        # Check if item idx fits in all remaining capacities
        w_idx = weights[:, idx]
        if np.all(w_idx <= remaining_cap):
            # Select it
            remaining_cap -= w_idx
            selected_items.append(idx)
            heuristic[idx] = total_prizes_per_item[idx]
        else:
            # It didn't fit. 
            # Why? Some constraint(s) are too tight.
            # We assign a lower score. 
            # The score should reflect that this item is "blocked" by the current set.
            # A simple penalty: score = prize * (min remaining capacity ratio)
            # This is complex. Let's just give a base score that is lower than selected ones.
            # But we want to preserve relative ranking among unselected items.
            # Let's assign score = base_priority[idx] * 0.1 (arbitrary damping)
            # Or better: keep the base_priority but scale it down so selected items dominate.
            heuristic[idx] = base_priority[idx]
    
    # Now, we have a binary-like distinction (selected vs not).
    # But we want a smooth gradient.
    
    # Let's refine: 
    # The heuristic value h_i should be high if item i is likely to be part of a good solution.
    # In ACO, the transition probability is P(i) = (tau_i * h_i) / sum(tau_j * h_j).
    # If h_i is very high for core items, ants will focus on them.
    
    # Let's normalize the scores.
    # Selected items have score = prize_sum.
    # Unselected items have score = base_priority.
    # These scales are different. prize_sum is ~2.5, base_priority is ~5.
    # This might make unselected items more attractive if their density is high.
    # We want selected items to be the "guiding stars".
    
    # Let's rescale unselected items to be slightly lower than the minimum selected item prize.
    if len(selected_items) > 0:
        min_selected_prize = np.min([total_prizes_per_item[i] for i in selected_items])
        # Scale unselected items so their max is ~0.9 * min_selected_prize
        unselected_mask = np.array([i not in selected_items for i in range(100)])
        if np.any(unselected_mask):
            max_unselected_score = np.max(heuristic[unselected_mask])
            if max_unselected_score > 0:
                scale_factor = (0.9 * min_selected_prize) / max_unselected_score
                heuristic[unselected_mask] *= scale_factor
    else:
        # No items selected? (Shouldn't happen with cap 25 and 100 items)
        heuristic = base_priority

    # Ensure positivity
    min_h = np.min(heuristic)
    if min_h < 0:
        heuristic = heuristic - min_h + 1e-6
        
    # Final safety: replace non-finite or non-positive with 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Time check
    if time.time() - start_time > 290:
        # Fallback
        total_prizes = np.sum(prizes, axis=0)
        total_weights = np.sum(weights, axis=0)
        heuristic = total_prizes / (total_weights + 1e-9)
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
