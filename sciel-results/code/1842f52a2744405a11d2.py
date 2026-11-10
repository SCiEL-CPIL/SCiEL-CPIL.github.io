"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: select the next item to add.
    
    Strategy: Greedy with 1-step Lookahead.
    For each feasible item, estimate the total value if we pick it,
    followed by a simple greedy density-based fill of the remaining capacity.
    Select the item with the highest estimated total value.
    """
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    capacity = float(remaining_capacity)
    
    # Check if any items are feasible
    feasible_mask = weights <= capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    # Prepare data for feasible items
    # We need to evaluate each feasible item as a candidate for the "next" pick.
    # For each candidate, we simulate a greedy fill of the rest.
    
    # Pre-compute densities for all feasible items to sort them once
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    
    # Avoid division by zero for density
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(f_weights > 0, f_values / f_weights, np.inf)
    
    # Sort indices by density descending
    # We need the global indices of the feasible items sorted by density
    sorted_local_indices = np.argsort(densities)[::-1]
    
    best_score = -1.0
    best_global_idx = -1
    
    # Iterate over each feasible item as a potential "next" item
    for local_idx in range(len(f_weights)):
        cand_w = f_weights[local_idx]
        cand_v = f_values[local_idx]
        
        rem_cap = capacity - cand_w
        
        # Simulate greedy fill with the remaining items
        # We iterate through the density-sorted list
        # If the item is the candidate, we skip it
        # If it fits, we take it
        sim_val = cand_v
        current_cap = rem_cap
        
        for s_idx in sorted_local_indices:
            if s_idx == local_idx:
                continue
            w = f_weights[s_idx]
            if w <= current_cap:
                sim_val += f_values[s_idx]
                current_cap -= w
        
        if sim_val > best_score:
            best_score = sim_val
            best_global_idx = int(feasible_indices[local_idx])
            
    return best_global_idx
# EVOLVE-BLOCK-END
