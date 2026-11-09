import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    # Filter feasible items first? 
    # The function returns index into remaining arrays.
    # If an item is infeasible (w > c), it cannot be picked.
    # We should only consider feasible items as candidates for the "next" pick.
    
    # 1. Sort indices by density descending
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    densities_safe = np.nan_to_num(densities, nan=0.0, posinf=1e18, neginf=0.0)
    sorted_indices = np.argsort(-densities_safe)
    
    best_score = -1
    best_idx = -1
    
    # Pre-compute lists for faster access?
    # w_sorted = w[sorted_indices]
    # v_sorted = v[sorted_indices]
    # But we need to skip 'i'.
    
    for i in range(n):
        wi = w[i]
        if wi > c:
            continue
            
        vi = v[i]
        c_new = c - wi
        
        # Estimate value of remaining items using 0/1 Greedy
        greedy_val = 0.0
        cap_left = c_new
        
        for idx in sorted_indices:
            if idx == i:
                continue
                
            wj = w[idx]
            vj = v[idx]
            
            if wj <= cap_left:
                greedy_val += vj
                cap_left -= wj
            else:
                # Greedy heuristic stops if next item doesn't fit
                # It does NOT take a fraction
                break
                
        total_score = vi + greedy_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            
    return best_idx
