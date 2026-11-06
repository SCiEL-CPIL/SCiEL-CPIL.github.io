import numpy as np
import time

_start_time = time.time()

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Deadline guard
    if time.time() - _start_time > 110:
        # Fallback to simple density
        # Ensure no division by zero
        w = np.asarray(remaining_weights)
        v = np.asarray(remaining_values)
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v / w
        return int(np.argmax(densities))

    w = remaining_weights
    v = remaining_values
    
    # Convert to lists for faster access in Python loop
    w_list = w.tolist()
    v_list = v.tolist()
    
    # Calculate densities
    # Handle zero weights just in case
    densities = []
    for i in range(n):
        if w_list[i] == 0:
            densities.append(float('inf'))
        else:
            densities.append(v_list[i] / w_list[i])
            
    # Sort indices by density descending
    # Create list of (density, index) and sort
    indexed_densities = [(d, i) for i, d in enumerate(densities)]
    indexed_densities.sort(key=lambda x: x[0], reverse=True)
    sorted_indices = [i for d, i in indexed_densities]
    
    # Select candidates
    # Top 10 by density
    cand_set = set(sorted_indices[:10])
    # Top 5 by value
    # Sort by value
    indexed_values = [(v_list[i], i) for i in range(n)]
    indexed_values.sort(key=lambda x: x[0], reverse=True)
    val_indices = [i for val, i in indexed_values[:5]]
    cand_set.update(val_indices)
    
    candidates = list(cand_set)
    
    best_val = -1
    best_idx = -1
    
    for c in candidates:
        wc = w_list[c]
        vc = v_list[c]
        
        cap = remaining_capacity - wc
        if cap < 0:
            continue
            
        val = vc
        
        for idx in sorted_indices:
            if idx == c:
                continue
            wi = w_list[idx]
            vi = v_list[idx]
            
            if wi <= cap:
                val += vi
                cap -= wi
                if cap <= 0:
                    break
            else:
                # Take fraction
                if wi > 0:
                    val += vi * (cap / wi)
                cap = 0
                break
        
        if val > best_val:
            best_val = val
            best_idx = c
            
    if best_idx == -1:
        # Fallback
        # Find max density index from sorted_indices
        return sorted_indices[0]
        
    return best_idx
