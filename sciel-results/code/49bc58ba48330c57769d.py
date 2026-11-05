import itertools
import sys

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks and compute their coverage masks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    block_masks = [0] * n_k_blocks
    
    for i, block in enumerate(k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # 3. Greedy Set Cover
    uncovered = (1 << n_t_subs) - 1
    selected_block_indices = []
    
    while uncovered:
        best_idx = -1
        max_gain = 0
        
        for i in range(n_k_blocks):
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                
        if best_idx == -1:
            break
            
        selected_block_indices.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    # 4. Iterative Local Search: Redundancy Removal
    # Repeatedly find and remove any block that is redundant (has no unique coverage)
    # This handles chains of redundancy (removing A makes B redundant, etc.)
    
    sol_indices = selected_block_indices
    
    while True:
        if not sol_indices:
            break
            
        # Calculate coverage counts for all t-subsets
        counts = [0] * n_t_subs
        for idx in sol_indices:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                m ^= lsb
                
        # Find one redundant block
        redundant_idx = -1
        for idx in sol_indices:
            m = block_masks[idx]
            essential = False
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    essential = True
                    break
                m ^= lsb
            if not essential:
                redundant_idx = idx
                break
                
        if redundant_idx == -1:
            break
            
        # Remove the redundant block
        sol_indices.remove(redundant_idx)
        
    # Convert indices back to blocks
    return [list(k_blocks[i]) for i in sol_indices]
