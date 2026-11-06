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
    
    # Calculate Lagrangian multiplier estimate (average density of feasible items)
    # We only consider feasible items for the lambda estimate to be robust
    w_feas = remaining_weights[feasible_indices]
    v_feas = remaining_values[feasible_indices]
    
    sum_w = np.sum(w_feas)
    sum_v = np.sum(v_feas)
    
    if sum_w == 0:
        return feasible_indices[0] # Edge case, pick first
        
    lambda_est = sum_v / sum_w
    
    # Calculate scores: v_i - lambda * w_i
    # This measures how much better the item is than the "average" item
    scores = v_feas - lambda_est * w_feas
    
    # Find the item with the highest score
    best_local_idx = np.argmax(scores)
    best_global_idx = feasible_indices[best_local_idx]
    
    return best_global_idx
