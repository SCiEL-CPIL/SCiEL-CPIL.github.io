"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights)
    values = np.asarray(remaining_values)

    # Identify feasible items
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1

    # Calculate densities for all items
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, np.inf)
    
    # Sort all items by density descending
    sorted_indices = np.argsort(-densities)
    
    best_score = -1
    best_index = -1
    
    # For each feasible item, estimate the total value if we pick it now.
    # Estimate = value_of_item + LP_Relaxation_Bound(remaining_items, remaining_capacity)
    
    for i in feasible_indices:
        current_val = values[i]
        rem_cap = remaining_capacity - weights[i]
        
        # Calculate LP bound for the rest
        # Iterate through items sorted by density, skipping i
        lp_bound = 0.0
        current_cap = rem_cap
        
        for j in sorted_indices:
            if j == i:
                continue
            if weights[j] <= current_cap:
                current_cap -= weights[j]
                lp_bound += values[j]
            else:
                # Take fraction
                if weights[j] > 0:
                    fraction = current_cap / weights[j]
                    lp_bound += values[j] * fraction
                current_cap = 0
                break
                
        total_score = current_val + lp_bound
        
        if total_score > best_score:
            best_score = total_score
            best_index = i
            
    return int(best_index)
# EVOLVE-BLOCK-END
