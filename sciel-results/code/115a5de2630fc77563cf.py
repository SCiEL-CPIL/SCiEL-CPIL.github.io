import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    # Check for feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    # Heuristic:
    # 1. If there are "large" items (weight > capacity / 2), 
    #    they consume most of the space. We should pick the one with max Value.
    # 2. Otherwise, pick the item with max Density (Value / Weight).
    
    weights = remaining_weights[feasible_indices]
    values = remaining_values[feasible_indices]
    
    # Threshold: half of remaining capacity
    threshold = remaining_capacity * 0.5
    
    large_mask = weights > threshold
    
    if np.any(large_mask):
        # Among large feasible items, pick max value
        large_indices = feasible_indices[large_mask]
        large_values = values[large_mask]
        best_local_idx = np.argmax(large_values)
        return large_indices[best_local_idx]
    else:
        # No large items. Pick max density.
        # Avoid division by zero (weights should be > 0, but check just in case)
        densities = values / weights
        best_local_idx = np.argmax(densities)
        return feasible_indices[best_local_idx]
