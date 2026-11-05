import itertools

def construct(v, k, t):
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    
    # Precompute bitmasks
    block_bitmasks = []
    for block in all_k_subsets:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subsets_to_idx[t_sub]
            mask |= (1 << idx)
        block_bitmasks.append(mask)
        
    all_ones = (1 << num_t_subsets) - 1
    uncovered_mask = all_ones
    selected_indices = []
    
    while uncovered_mask:
        best_idx = -1
        best_count = -1
        
        for i, mask in enumerate(block_bitmasks):
            count = (uncovered_mask & mask).bit_count()
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        uncovered_mask &= ~block_bitmasks[best_idx]
        
    # Redundancy removal
    changed = True
    while changed:
        changed = False
        new_indices = []
        for i, idx in enumerate(selected_indices):
            # Check if block idx is redundant given the others in selected_indices
            # Union of all other selected blocks
            union_others = 0
            for j, other_idx in enumerate(selected_indices):
                if i != j:
                    union_others |= block_bitmasks[other_idx]
            
            # If union_others covers everything, block idx is redundant
            if (union_others & all_ones) == all_ones:
                changed = True
            else:
                new_indices.append(idx)
        selected_indices = new_indices
        
    return [list(all_k_subsets[i]) for i in selected_indices]
