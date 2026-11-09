import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    start_time = time.time()
    deadline = 145.0
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    
    n = len(weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)
    
    if f_n == 0:
        return -1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(f_weights > 0, f_values / f_weights, np.inf)
    
    # Sort indices by density descending
    global_sort_idx = np.argsort(-densities)
    
    # Candidate set for first item
    K1 = min(f_n, 40)
    idx_by_density = global_sort_idx
    idx_by_value = np.argsort(-f_values)
    idx_by_weight = np.argsort(f_weights)
    
    candidates = np.unique(np.concatenate([
        idx_by_density[:K1],
        idx_by_value[:10],
        idx_by_weight[:10]
    ]))
    
    if f_n <= 50:
        candidates = np.arange(f_n)
        
    best_score = -1
    best_first_idx = -1
    
    # Helper to compute fractional fill value
    # skip_mask is a boolean array of size f_n, True if item is skipped
    def get_frac_fill(cap, skip_mask):
        if cap <= 0:
            return 0.0
        total_val = 0.0
        rem = cap
        for idx in global_sort_idx:
            if skip_mask[idx]:
                continue
            if rem <= 0:
                break
            w = f_weights[idx]
            v = f_values[idx]
            if w <= rem:
                rem -= w
                total_val += v
            else:
                total_val += v * (rem / w)
                rem = 0
                break
        return total_val

    K2 = min(f_n - 1, 20)
    
    for i in candidates:
        if time.time() - start_time > deadline - 5.0:
            break
            
        w_i = f_weights[i]
        v_i = f_values[i]
        
        if w_i > remaining_capacity:
            continue
            
        rem_cap_after_i = remaining_capacity - w_i
        
        # Find second candidates: top K2 by density from remaining feasible items
        second_cands = []
        for idx in global_sort_idx:
            if idx == i:
                continue
            if len(second_cands) >= K2:
                break
            if f_weights[idx] <= rem_cap_after_i:
                second_cands.append(idx)
                
        # Case 1: No second item fits or we choose to stop after i
        # Evaluate i alone with fractional fill of rest
        skip_mask = np.zeros(f_n, dtype=bool)
        skip_mask[i] = True
        score_alone = v_i + get_frac_fill(rem_cap_after_i, skip_mask)
        
        if score_alone > best_score:
            best_score = score_alone
            best_first_idx = i
            
        # Case 2: Evaluate pairs (i, j)
        for j in second_cands:
            if time.time() - start_time > deadline - 5.0:
                break
                
            w_j = f_weights[j]
            v_j = f_values[j]
            
            rem_cap_after_j = rem_cap_after_i - w_j
            
            if rem_cap_after_j < 0:
                continue
                
            skip_mask[j] = True
            fill_val = get_frac_fill(rem_cap_after_j, skip_mask)
            skip_mask[j] = False # Reset for next iteration? No, we can just recreate or use a set.
            
            # Actually, creating a new mask each time is safer and fast enough for N=100
            # But let's optimize: use a set for skips in the inner loop?
            # The get_frac_fill uses a mask. Let's stick to mask but create it fresh?
            # No, the mask is size 100. Creating it is cheap.
            
            # Let's re-implement get_frac_fill to take a set of skips for simplicity in the inner loop
            # Or just pass the indices.
            
            # Let's optimize the inner loop by inlining the fractional fill logic
            # to avoid function call overhead and mask creation.
            
            rem = rem_cap_after_j
            fill_val = 0.0
            for idx in global_sort_idx:
                if idx == i or idx == j:
                    continue
                if rem <= 0:
                    break
                w = f_weights[idx]
                v = f_values[idx]
                if w <= rem:
                    rem -= w
                    fill_val += v
                else:
                    fill_val += v * (rem / w)
                    rem = 0
                    break
                    
            score = v_i + v_j + fill_val
            
            if score > best_score:
                best_score = score
                best_first_idx = i
                
    if best_first_idx == -1:
        return -1
        
    return int(feasible_indices[best_first_idx])
