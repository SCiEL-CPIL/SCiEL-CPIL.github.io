    ids_local = ids
    freq_local = freq
    best_dist = best_distinct
    best_n_val = best_n
    
    for n in range(1001, 1000001):
        start_idx = n - 1000
        out_idx = start_idx - 1
        in_idx = start_idx + W - 1
        
        out_id = ids_local[out_idx]
        in_id = ids_local[in_idx]
        
        freq_local[out_id] -= 1
        if freq_local[out_id] == 0:
            best_dist -= 1
            
        freq_local[in_id] += 1
        if freq_local[in_id] == 1:
            best_dist += 1
            
        if best_dist < best_distinct:
            best_distinct = best_dist
            best_n = n
            if best_dist == 1:
                break
