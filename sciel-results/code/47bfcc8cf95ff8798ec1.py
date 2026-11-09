import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    # Pre-compute density-sorted indices for the entire set
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    densities_safe = np.nan_to_num(densities, nan=0.0, posinf=1e18, neginf=0.0)
    sorted_indices = np.argsort(-densities_safe)
    
    best_score = -1.0
    best_idx = -1
    
    for i in range(n):
        wi = w[i]
        if wi > c:
            continue
            
        vi = v[i]
        cap_left = c - wi
        
        # Calculate Fractional Knapsack Upper Bound for remaining items
        # excluding item i
        rem_val = 0.0
        
        for idx in sorted_indices:
            if idx == i:
                continue
                
            wj = w[idx]
            vj = v[idx]
            
            if wj <= cap_left:
                rem_val += vj
                cap_left -= wj
            else:
                # Take fraction
                if wj > 0:
                    rem_val += vj * (cap_left / wj)
                cap_left = 0
                break
                
        total_score = vi + rem_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            
    return best_idx
