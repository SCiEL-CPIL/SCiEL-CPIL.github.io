import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    n = len(w)
    
    if n == 0:
        return -1
        
    # Identify feasible items
    feasible_mask = w <= remaining_capacity
    
    if not np.any(feasible_mask):
        return -1
        
    # Get indices of feasible items
    feasible_indices = np.where(feasible_mask)[0]
    
    # Calculate densities for feasible items
    # Avoid division by zero (weights should be > 0, but safe check)
    feasible_w = w[feasible_indices]
    feasible_v = v[feasible_indices]
    
    # Filter out zero weights just in case, though problem states weights > 0
    valid_mask = feasible_w > 0
    if not np.any(valid_mask):
        return -1
        
    feasible_indices = feasible_indices[valid_mask]
    feasible_w = feasible_w[valid_mask]
    feasible_v = feasible_v[valid_mask]
    
    densities = feasible_v / feasible_w
    
    # The item with the highest density is the one that would be selected first
    # in the optimal fractional knapsack solution.
    best_local_idx = np.argmax(densities)
    best_global_idx = feasible_indices[best_local_idx]
    
    return int(best_global_idx)
