import itertools
import sys

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute masks for each k-block
    block_masks = [0] * n_k_blocks
    # Precompute sub_to_blocks: index of t-subset -> list of block indices
    sub_to_blocks = [[] for _ in range(n_t_subs)]
    
    for i, block in enumerate(k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
            sub_to_blocks[idx].append(i)
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
        
    # 4. Local Search
    if not selected_block_indices:
        return []
        
    # Helper to update counts
    def update_counts(indices, delta):
        counts = [0] * n_t_subs
        for idx in indices:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += delta
                m ^= lsb
        return counts

    # Initial counts
    counts = update_counts(selected_block_indices, 1)
    
    # Helper to check if a block is redundant
    def is_redundant(idx, counts):
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                return False
            m ^= lsb
        return True

    # Helper to remove redundant blocks
    def remove_redundant(selected, counts):
        new_selected = []
        for idx in selected:
            if is_redundant(idx, counts):
                # Remove idx
                m = block_masks[idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    counts[bit_idx] -= 1
                    m ^= lsb
            else:
                new_selected.append(idx)
        return new_selected

    # Local Search Loop
    improved = True
    while improved:
        improved = False
        
        # Phase 1: Remove redundant blocks
        new_selected = remove_redundant(selected_block_indices, counts)
        if len(new_selected) < len(selected_block_indices):
            selected_block_indices = new_selected
            improved = True
            # Counts are already updated in remove_redundant
            
        # Phase 2: Try swaps
        # Iterate over a copy of the list to avoid modification issues during iteration
        # We need to be careful with indices changing.
        # It's better to iterate by index and check if still in list.
        
        current_len = len(selected_block_indices)
        for i in range(current_len):
            if i >= len(selected_block_indices):
                continue
            b_i = selected_block_indices[i]
            
            # Temporarily remove b_i
            m_bi = block_masks[b_i]
            # Check if it's redundant (should be caught in Phase 1, but maybe counts changed?)
            # Actually, if we are in Phase 2, Phase 1 just ran. So b_i is not redundant.
            
            # Identify uncovered subsets if b_i is removed
            # These are subsets in b_i where counts[s] == 1
            U = []
            m_temp = m_bi
            while m_temp:
                lsb = m_temp & -m_temp
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    U.append(bit_idx)
                m_temp ^= lsb
                
            if not U:
                # Should have been removed in Phase 1
                continue
                
            # Decrement counts for b_i
            m_dec = m_bi
            while m_dec:
                lsb = m_dec & -m_dec
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] -= 1
                m_dec ^= lsb
                
            # Remove b_i from selected
            selected_block_indices.pop(i)
            # Adjust loop index? 
            # Since we pop, indices shift. 
            # It's safer to restart the loop or handle carefully.
            # Let's just break and restart the while loop if we make a change.
            
            # Find candidates for b_new that cover all U
            # Intersection of sub_to_blocks[s] for s in U
            # Start with the list for the rarest subset in U
            # Sort U by frequency of sub_to_blocks[s]
            U_sorted = sorted(U, key=lambda s: len(sub_to_blocks[s]))
            if not U_sorted:
                # Should not happen
                pass
            
            candidates = set(sub_to_blocks[U_sorted[0]])
            for s in U_sorted[1:]:
                candidates &= set(sub_to_blocks[s])
                if not candidates:
                    break
            
            swap_made = False
            for c in candidates:
                # c must not be in selected_block_indices
                if c in selected_block_indices:
                    continue
                    
                # Tentatively add c
                m_c = block_masks[c]
                m_inc = m_c
                while m_inc:
                    lsb = m_inc & -m_inc
                    bit_idx = lsb.bit_length() - 1
                    counts[bit_idx] += 1
                    m_inc ^= lsb
                    
                selected_block_indices.append(c)
                
                # Check if this addition allows removal of any other block
                # We check if any block in selected_block_indices is now redundant
                # Note: c is now in the list.
                redundant_found = False
                new_selected = []
                for idx in selected_block_indices:
                    if is_redundant(idx, counts):
                        # Remove idx
                        m_rm = block_masks[idx]
                        while m_rm:
                            lsb = m_rm & -m_rm
                            bit_idx = lsb.bit_length() - 1
                            counts[bit_idx] -= 1
                            m_rm ^= lsb
                        redundant_found = True
                    else:
                        new_selected.append(idx)
                        
                if redundant_found:
                    selected_block_indices = new_selected
                    swap_made = True
                    improved = True
                    break # Restart the whole while loop
                else:
                    # Revert
                    selected_block_indices.pop() # Remove c
                    m_dec_c = m_c
                    while m_dec_c:
                        lsb = m_dec_c & -m_dec_c
                        bit_idx = lsb.bit_length() - 1
                        counts[bit_idx] -= 1
                        m_dec_c ^= lsb
                    
            # Revert removal of b_i if no swap was made
            if not swap_made:
                selected_block_indices.insert(i, b_i)
                m_inc_bi = m_bi
                while m_inc_bi:
                    lsb = m_inc_bi & -m_inc_bi
                    bit_idx = lsb.bit_length() - 1
                    counts[bit_idx] += 1
                    m_inc_bi ^= lsb

    return [list(k_blocks[i]) for i in selected_block_indices]
