import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    W = np.array(remaining_weights)
    V = np.array(remaining_values)
    C = remaining_capacity
    
    # Check for feasible items
    feasible_mask = W <= C
    if not np.any(feasible_mask):
        return -1
        
    # Calculate densities
    # Handle zero weights (though task implies weights > 0, good practice)
    densities = np.divide(V, W, out=np.zeros_like(V, dtype=float), where=W!=0)
    
    # Sort indices by density descending to find the marginal item
    sorted_indices = np.argsort(densities)[::-1]
    
    cum_weight = 0.0
    lambda_val = 0.0
    
    for idx in sorted_indices:
        if cum_weight + W[idx] <= C:
            cum_weight += W[idx]
        else:
            lambda_val = densities[idx]
            break
    else:
        # If all items fit, the marginal value of capacity is 0
        lambda_val = 0.0

    feasible_indices = np.where(feasible_mask)[0]
    
    # Calculate Lagrangian score for each feasible item: v_i - lambda * w_i
    # This represents the "net benefit" of the item given the shadow price of capacity
    scores = V[feasible_indices] - lambda_val * W[feasible_indices]
    
    # Select the item with the highest score
    best_local_idx = np.argmax(scores)
    best_global_idx = feasible_indices[best_local_idx]
    
    return int(best_global_idx)
