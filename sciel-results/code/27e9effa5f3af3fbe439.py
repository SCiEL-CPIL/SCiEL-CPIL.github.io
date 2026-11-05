import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # Find indices of items that fit in the remaining capacity
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # Calculate densities for all items to use for sorting
    # Handle zero weight (though unlikely in knapsack, safe to guard)
    weights = np.asarray(remaining_weights)
    values = np.asarray(remaining_values)
    
    # To avoid division by zero, use a large number for weight 0
    safe_weights = np.where(weights == 0, 1e-9, weights)
    densities = values / safe_weights
    
    # Sort indices by density descending
    # We need to consider all items in the 'remaining' set, not just feasible ones?
    # The greedy fill will only pick items that fit.
    # So we sort the entire remaining set by density.
    sorted_indices = np.argsort(-densities)
    
    best_total_value = -1
    best_item_index = -1
    
    # For each feasible candidate item, simulate picking it and then filling greedily
    for i in feasible_indices:
        # Value if we pick item i
        current_value = values[i]
        remaining_cap = remaining_capacity - weights[i]
        
        # Fill the rest greedily from the sorted list
        # Skip item i
        for j in sorted_indices:
            if j == i:
                continue
            if weights[j] <= remaining_cap:
                current_value += values[j]
                remaining_cap -= weights[j]
        
        if current_value > best_total_value:
            best_total_value = current_value
            best_item_index = i
            
    return int(best_item_index)
