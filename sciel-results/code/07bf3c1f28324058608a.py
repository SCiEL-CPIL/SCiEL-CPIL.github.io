import numpy as np
import time

def _fractional_knapsack_value(capacity, weights, values):
    """
    Calculate the optimal value of the fractional knapsack problem.
    weights and values are arrays of the same length.
    Returns the total fractional value.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
    
    # Calculate densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    # Handle zero weights (infinite density, take all)
    densities[weights == 0] = np.inf
    
    # Sort by density descending
    order = np.argsort(-densities)
    
    total_value = 0.0
    remaining_cap = capacity
    
    for idx in order:
        w = weights[idx]
        v = values[idx]
        
        if w <= remaining_cap:
            total_value += v
            remaining_cap -= w
        else:
            # Take fraction
            if w > 0:
                total_value += v * (remaining_cap / w)
            break
            
    return total_value

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Find feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    feasible_idx = np.where(feasible_mask)[0]
    
    if len(feasible_idx) == 0:
        return -1
        
    # If there is only one feasible item, pick it
    if len(feasible_idx) == 1:
        return int(feasible_idx[0])
        
    best_total = -1.0
    best_idx = -1
    
    w = remaining_weights
    v = remaining_values
    
    # Pre-compute a mask for remaining items for each candidate? 
    # Instead of creating new arrays, we can pass the original arrays and a mask to the fractional solver?
    # Or just create the sub-arrays. N=100 is small.
    
    for idx in feasible_idx:
        w_k = w[idx]
        v_k = v[idx]
        
        new_cap = remaining_capacity - w_k
        
        # Prepare remaining arrays excluding idx
        # Using boolean masking is efficient enough for N=100
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        new_w = w[mask]
        new_v = v[mask]
        
        # Calculate fractional value of the rest
        frac_val = _fractional_knapsack_value(new_cap, new_w, new_v)
        
        total_estimate = v_k + frac_val
        
        if total_estimate > best_total:
            best_total = total_estimate
            best_idx = idx
            
    return int(best_idx)
