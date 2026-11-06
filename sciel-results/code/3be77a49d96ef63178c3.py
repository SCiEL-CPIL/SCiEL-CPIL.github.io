import numpy as np
import time

def _greedy_pack(capacity, weights, values):
    """
    Greedily packs items by density.
    Returns total value.
    """
    if capacity <= 1e-9 or len(weights) == 0:
        return 0.0
        
    # Calculate density
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(weights > 1e-9, values / weights, np.inf)
        
    # Sort indices by density descending
    # np.argsort returns indices that would sort the array
    # We want descending, so sort by -density
    indices = np.argsort(-density)
    
    total_val = 0.0
    curr_cap = capacity
    
    for idx in indices:
        w = weights[idx]
        if w <= curr_cap:
            total_val += values[idx]
            curr_cap -= w
            if curr_cap <= 1e-9:
                break
                
    return total_val

def _two_step_lookahead(capacity, weights, values, max_candidates=20):
    """
    Evaluates the best value obtainable by picking one item, then greedily filling the rest.
    To speed up, only considers the top 'max_candidates' items by density as the first pick.
    """
    n = len(weights)
    if n == 0 or capacity <= 1e-9:
        return 0.0
        
    # Filter feasible items
    feasible_mask = weights <= capacity
    if not np.any(feasible_mask):
        return 0.0
        
    w_feas = weights[feasible_mask]
    v_feas = values[feasible_mask]
    
    n_feas = len(w_feas)
    if n_feas == 0:
        return 0.0
        
    # Calculate density for feasible items
    with np.errstate(divide='ignore', invalid='ignore'):
        density = np.where(w_feas > 1e-9, v_feas / w_feas, np.inf)
        
    # Get top candidates by density
    # If n_feas is small, check all
    k = min(max_candidates, n_feas)
    top_indices = np.argsort(-density)[:k]
    
    best_val = 0.0
    
    for idx in top_indices:
        w_i = w_feas[idx]
        v_i = v_feas[idx]
        
        rem_cap = capacity - w_i
        if rem_cap <= 1e-9:
            # Only this item fits
            if v_i > best_val:
                best_val = v_i
            continue
            
        # Create mask to exclude idx
        mask = np.ones(n_feas, dtype=bool)
        mask[idx] = False
        
        w_rem = w_feas[mask]
        v_rem = v_feas[mask]
        
        # Greedy fill the rest
        val_rest = _greedy_pack(rem_cap, w_rem, v_rem)
        total_val = v_i + val_rest
        
        if total_val > best_val:
            best_val = total_val
            
    return best_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    
    # Quick check: if no item fits, return -1
    if np.all(w > remaining_capacity):
        return -1

    best_idx = -1
    best_total_val = -1.0
    
    # Sort items by density descending to process promising ones first?
    # Or just iterate all.
    # To allow early stopping or pruning, we might want an upper bound.
    # Upper bound for item i: v_i + sum(v_j for j != i, w_j <= rem_cap)
    # This is expensive.
    
    # We will iterate through all feasible items.
    # To speed up, we can pre-calculate the "greedy fill" for the whole set? No.
    
    # Optimization: 
    # If n is large, the 2-step lookahead for the remainder is O(N * N log N).
    # Doing this for all N items is O(N^2 * N log N) = O(N^3 log N).
    # For N=100, 10^6 * 7 = 7*10^6. This is fine.
    
    # However, we can limit the "max_candidates" in _two_step_lookahead if time is tight.
    # Let's start with a moderate limit.
    
    max_candidates_for_remainder = 20
    
    for i in range(n):
        if time.time() > deadline:
            break
            
        w_i = w[i]
        if w_i > remaining_capacity:
            continue
            
        v_i = v[i]
        
        rem_cap = remaining_capacity - w_i
        
        # Prepare remainder arrays
        # Exclude i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        # Filter by fit in rem_cap
        fit_mask = w <= rem_cap
        valid_mask = mask & fit_mask
        
        if np.any(valid_mask):
            w_rem = w[valid_mask]
            v_rem = v[valid_mask]
            
            # Estimate future value using 2-step lookahead (pick best next, then greedy)
            # If rem_cap is small or few items, this is fast.
            future_val = _two_step_lookahead(rem_cap, w_rem, v_rem, max_candidates=max_candidates_for_remainder)
        else:
            future_val = 0.0
            
        total_val = v_i + future_val
        
        if total_val > best_total_val:
            best_total_val = total_val
            best_idx = i
            
    return int(best_idx)
