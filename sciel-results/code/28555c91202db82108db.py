"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Returns attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Strategy: Greedy Simulation with Conflict Resolution.
    1. Compute a base efficiency score for each item (prize / max_weight).
    2. Simulate a greedy packing process to identify a "core" set of items that 
       fit together well.
    3. Items in the core set get boosted scores.
    4. Items that are "blocked" (high weight, low prize, or conflict with core) 
       get suppressed.
    5. This guides the ACO to focus on items that form a feasible, high-value 
       combination rather than just individually efficient items.
    """
    start_time = time.time()
    
    # Dimensions
    n_constraints, n_items = prizes.shape
    capacity = 25.0
    
    # 1. Base Efficiency: Prize per unit of "bottleneck" weight
    # Use max weight across constraints as a proxy for "cost"
    max_weight = weights.max(axis=0)
    total_prize = prizes.sum(axis=0)
    
    # Avoid division by zero
    max_weight_safe = np.where(max_weight < 1e-9, 1e-9, max_weight)
    base_eff = total_prize / max_weight_safe
    
    # 2. Simulate Greedy Packing to find a "good" subset
    # We simulate a simple greedy algorithm that picks items in order of base_eff
    # until no more items fit. This identifies which items are "compatible".
    
    # Sort items by base efficiency (descending)
    order = np.argsort(-base_eff)
    
    # Remaining capacities
    rem_cap = np.full(n_constraints, capacity)
    
    # Track which items were selected in this greedy simulation
    selected_mask = np.zeros(n_items, dtype=bool)
    selected_count = 0
    max_selectable = 20  # Heuristic limit to avoid O(n^2) in worst case, though n=100 is small
    
    for idx in order:
        if selected_count >= max_selectable:
            break
        if time.time() - start_time > 290:  # Safety margin
            break
            
        # Check if item fits
        w = weights[:, idx]
        if np.all(w <= rem_cap):
            # Select it
            rem_cap -= w
            selected_mask[idx] = True
            selected_count += 1
            
    # 3. Construct Heuristic Scores
    # Items in the greedy core are highly attractive.
    # Items not in the core are less attractive, but we keep a gradient based on 
    # their potential to replace core items or fit in remaining space.
    
    # Score for selected items: Boosted by their actual contribution in the greedy set
    # Score for non-selected items: Based on how close they are to fitting in the 
    # "gaps" of the greedy solution, or just their base efficiency scaled down.
    
    # Let's refine:
    # High score for items that were selected.
    # Lower score for items that were not selected but have high base_eff (potential swaps).
    # Very low score for items that are heavy and low prize.
    
    # Normalize base_eff to [0, 1] for mixing
    max_base = base_eff.max()
    if max_base > 0:
        norm_base = base_eff / max_base
    else:
        norm_base = np.zeros(n_items)
        
    # Initialize heuristic
    heuristic = np.zeros(n_items)
    
    # For selected items: give them a high score, proportional to their efficiency
    heuristic[selected_mask] = 1.0 + norm_base[selected_mask]
    
    # For non-selected items: give them a lower score, but still reflect their efficiency
    # This allows the ACO to potentially find better combinations than the greedy one.
    heuristic[~selected_mask] = 0.1 + 0.5 * norm_base[~selected_mask]
    
    # 4. Conflict Suppression (Optional refinement)
    # If an item is very similar to a selected item (high weight in same constraints, 
    # lower prize), it might be redundant. 
    # However, simple greedy simulation already handles most "fit" issues.
    # Let's add a penalty for items that are "heavy" in constraints that are already 
    # tight in the greedy solution.
    
    # Tightness of constraints in the greedy solution
    tightness = 1.0 - (rem_cap / capacity)  # 0 = loose, 1 = full
    
    # For each non-selected item, check if it would have fit if we removed one item?
    # That's complex. Instead, penalize items that consume resources from tight constraints.
    
    # Weighted cost of non-selected items based on constraint tightness
    # cost[i] = sum(tightness[c] * weight[i,c])
    costs = weights.T @ tightness  # shape (n_items,)
    
    # Normalize costs
    max_cost = costs.max()
    if max_cost > 0:
        norm_cost = costs / max_cost
    else:
        norm_cost = np.zeros(n_items)
        
    # Apply penalty to non-selected items: if they are expensive in tight constraints, lower score
    penalty = 0.5 * norm_cost
    heuristic[~selected_mask] -= penalty[~selected_mask]
    
    # Ensure positivity
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Final check for time
    if time.time() - start_time > 295:
        return heuristic
    
    return heuristic
# EVOLVE-BLOCK-END
