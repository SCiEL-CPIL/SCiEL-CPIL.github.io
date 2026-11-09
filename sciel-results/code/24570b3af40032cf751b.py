import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    c = float(remaining_capacity)
    
    # Handle potential division by zero
    # If weight is 0, density is inf. It should always be picked if value > 0.
    # But standard knapsack assumes w > 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
        # Replace inf with a large number for sorting, or handle separately
        # If w==0 and v>0, it's a free value. 
        # If w==0 and v==0, it's useless.
        # Assuming standard w > 0.
        
    # Sort indices by density descending
    # np.nan_to_num to handle any nan from 0/0
    densities_safe = np.nan_to_num(densities, nan=0.0, posinf=1e18, neginf=0.0)
    sorted_indices = np.argsort(-densities_safe)
    
    best_score = -1
    best_idx = -1
    
    for i in range(n):
        wi = w[i]
        vi = v[i]
        
        # Check feasibility
        if wi > c:
            continue
            
        c_new = c - wi
        
        # Calculate fractional bound for remaining items (excluding i) with capacity c_new
        bound_val = 0.0
        cap_left = c_new
        
        for idx in sorted_indices:
            if idx == i:
                continue
                
            wj = w[idx]
            vj = v[idx]
            
            if wj <= cap_left:
                bound_val += vj
                cap_left -= wj
            else:
                if wj > 0:
                    bound_val += vj * (cap_left / wj)
                break
                
        total_score = vi + bound_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            
    return best_idx
