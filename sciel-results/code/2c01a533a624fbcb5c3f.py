import numpy as np

def _fractional_knapsack_value(capacity, weights, values):
    """
    Computes the optimal value of the fractional knapsack problem
    for the given items and capacity.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
        
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    
    # Filter out any zero or negative weights just in case
    valid = w > 0
    if not np.any(valid):
        return 0.0
    w = w[valid]
    v = v[valid]
    
    # Calculate densities
    densities = v / w
    
    # Sort by density descending
    order = np.argsort(-densities)
    w_sorted = w[order]
    v_sorted = v[order]
    
    total_value = 0.0
    remaining_cap = capacity
    
    for wi, vi in zip(w_sorted, v_sorted):
        if remaining_cap <= 0:
            break
        if wi <= remaining_cap:
            total_value += vi
            remaining_cap -= wi
        else:
            # Take fraction
            total_value += vi * (remaining_cap / wi)
            remaining_cap = 0
            break
            
    return total_value

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
        
    feasible_indices = np.where(feasible_mask)[0]
    
    best_score = -np.inf
    best_index = -1
    
    # Evaluate each feasible item by looking ahead with the fractional surrogate
    for idx in feasible_indices:
        # If we pick this item, we use w[idx] capacity and gain v[idx]
        # The rest of the capacity is filled optimally (fractionally) by the remaining items
        
        remaining_cap_after = remaining_capacity - w[idx]
        
        # Create arrays of the remaining items (excluding idx)
        # We need to be careful with indexing
        if idx == 0:
            w_rest = w[1:]
            v_rest = v[1:]
        elif idx == n - 1:
            w_rest = w[:-1]
            v_rest = v[:-1]
        else:
            w_rest = np.concatenate([w[:idx], w[idx+1:]])
            v_rest = np.concatenate([v[:idx], v[idx+1:]])
            
        # Calculate the fractional value of the rest
        fk_value_rest = _fractional_knapsack_value(remaining_cap_after, w_rest, v_rest)
        
        total_score = v[idx] + fk_value_rest
        
        if total_score > best_score:
            best_score = total_score
            best_index = idx
            
    return int(best_index)
