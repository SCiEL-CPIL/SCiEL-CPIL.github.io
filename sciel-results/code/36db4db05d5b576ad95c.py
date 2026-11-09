def generate_random_packing():
    # valid_indices is a set of indices of triples that are currently valid
    valid_indices = set(range(NUM_TRIPLES))
    random_list = list(valid_indices)
    random.shuffle(random_list)
    
    selected = []
    
    for idx in random_list:
        if idx not in valid_indices:
            continue
            
        m = TRIPLE_MASKS[idx]
        selected.append(list(ALL_TRIPLES[idx]))
        
        # Remove invalid triples from valid_indices
        # A triple u is invalid if it shares 2 or 3 points with m
        # i.e., (m & mask_u).bit_count() > 1
        to_remove = []
        for u_idx in valid_indices:
            if u_idx == idx:
                continue
            if (m & TRIPLE_MASKS[u_idx]).bit_count() > 1:
                to_remove.append(u_idx)
                
        for u_idx in to_remove:
            valid_indices.remove(u_idx)
            
        # Optional: if valid_indices is empty, break
        if not valid_indices:
            break
            
    return selected
