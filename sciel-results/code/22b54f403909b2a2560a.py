"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Convert to numpy arrays for convenience if not already
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)

    # Identify feasible items
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1

    # Pre-sort all items by density (value/weight) descending to use for simulation
    # We need to handle weight=0 case, but usually weights > 0. 
    # If weight is 0, density is inf.
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, np.inf)
    
    # Sort indices by density descending
    sorted_indices = np.argsort(-densities, kind='stable')
    
    # For efficiency, we only consider a subset of top candidates for the first pick
    # if there are too many feasible items. However, with 100 items, checking all is fine.
    # Let's stick to checking all feasible items but optimize the inner loop.
    
    best_value = -1
    best_index = -1
    
    # Pre-compute prefix sums for the sorted items to speed up greedy fill estimation?
    # Actually, the greedy fill depends on the specific remaining capacity after picking item i.
    # A simple greedy simulation is O(N^2) in worst case if we do it for every item.
    # With N=100, N^2 = 10,000 operations per call. The function is called up to 100 times.
    # Total ops ~ 1,000,000. This is well within 120s.
    
    # To make it faster, we can pre-sort the items and use a helper to simulate greedy fill.
    
    for i in feasible_indices:
        # Pick item i
        current_val = values[i]
        rem_cap = remaining_capacity - weights[i]
        
        # Greedily fill the rest with other items
        # We iterate through the pre-sorted list
        fill_val = 0.0
        current_cap = rem_cap
        
        for j in sorted_indices:
            if j == i:
                continue
            if weights[j] <= current_cap:
                current_cap -= weights[j]
                fill_val += values[j]
                if current_cap <= 0:
                    break
        
        total_val = current_val + fill_val
        
        if total_val > best_value:
            best_value = total_val
            best_index = i
            
    return int(best_index)
# EVOLVE-BLOCK-END
