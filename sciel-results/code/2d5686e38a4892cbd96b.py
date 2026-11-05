    import itertools
    import sys
    import random

    def construct(v, k, t):
        points = range(v)
        
        # 1. Enumerate all t-subsets
        t_subs = list(itertools.combinations(points, t))
        n_t_subs = len(t_subs)
        sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
        
        # 2. Enumerate all k-blocks and compute masks
        k_blocks = list(itertools.combinations(points, k))
        n_k_blocks = len(k_blocks)
        
        # Precompute masks
        block_masks = [0] * n_k_blocks
        for i, block in enumerate(k_blocks):
            mask = 0
            for t_sub in itertools.combinations(block, t):
                idx = sub_to_idx[t_sub]
                mask |= (1 << idx)
            block_masks[i] = mask
        
        # Helper to count bits
        def popcount(x):
            return x.bit_count()
            
        # 3. Greedy Construction with Weights
        # Weights for t-subsets. Initially 1.
        # We will use a weighted greedy to pick the best block.
        
        def greedy_construct(weights=None):
            if weights is None:
                weights = [1] * n_t_subs
            
            uncovered_mask = (1 << n_t_subs) - 1
            selected = []
            # We need to efficiently find the block with max weighted coverage.
            # Since n_k_blocks is ~75k, scanning all each time is O(75k * 335) ~ 25M.
            # In Python, this might be tight but doable if optimized.
            # However, we can just scan.
            
            while uncovered_mask:
                best_idx = -1
                max_score = -1
                for i in range(n_k_blocks):
                    # Calculate weighted score
                    # mask_i & uncovered_mask
                    # We need sum of weights for bits in (mask_i & uncovered_mask)
                    # This is expensive if done naively per bit.
                    # Optimization: If weights are all 1, just use popcount.
                    # If weights vary, we can precompute? No, weights change? 
                    # In standard weighted greedy for set cover, weights are static or updated.
                    # Let's stick to static weights for simplicity first, or dynamic.
                    # Dynamic: Increase weight of uncovered subsets.
                    
                    # For now, let's use popcount (weight=1) for the greedy step.
                    # We will add a local search phase to improve.
                    m = block_masks[i] & uncovered_mask
                    if m:
                        score = m.bit_count()
                        if score > max_score:
                            max_score = score
                            best_idx = i
                            
                if best_idx == -1:
                    break
                    
                selected.append(best_idx)
                uncovered_mask &= ~block_masks[best_idx]
                
            return selected
            
        # Initial Greedy
        selected = greedy_construct()
        
        # 4. Local Search: Delete Redundant Blocks
        def remove_redundants(selected):
            # Convert to set for fast lookup
            sel_set = set(selected)
            # Count coverage
            counts = [0] * n_t_subs
            for idx in selected:
                m = block_masks[idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    counts[bit_idx] += 1
                    m ^= lsb
                    
            new_selected = []
            for idx in selected:
                m = block_masks[idx]
                essential = False
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    if counts[bit_idx] == 1:
                        essential = True
                        break
                    m ^= lsb
                    
                if essential:
                    new_selected.append(idx)
                else:
                    # Decrement counts
                    m = block_masks[idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        counts[bit_idx] -= 1
                        m ^= lsb
                        
            return new_selected
            
        selected = remove_redundants(selected)
        
        # 5. Local Search: Swap to improve
        # Try to swap a block in S with a block not in S to enable further deletions.
        # This is a 1-step lookahead.
        
        def try_swap_improvement(selected):
            sel_set = set(selected)
            improved = True
            while improved:
                improved = False
                # Calculate current coverage counts
                counts = [0] * n_t_subs
                for idx in selected:
                    m = block_masks[idx]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        counts[bit_idx] += 1
                        m ^= lsb
                        
                for i, b_idx in enumerate(selected):
                    # Try removing b_idx
                    # What becomes uncovered?
                    # A subset s is uncovered if counts[s] == 1 and s is in b_idx.
                    # Let's find the set of subsets that become uncovered if we remove b_idx.
                    m_b = block_masks[b_idx]
                    uncovered_if_removed = 0
                    while m_b:
                        lsb = m_b & -m_b
                        bit_idx = lsb.bit_length() - 1
                        if counts[bit_idx] == 1:
                            uncovered_if_removed |= (1 << bit_idx)
                        m_b ^= lsb_b # Typo in thought, correct below
                        
                    # Correct logic:
                    uncovered_if_removed = 0
                    m_b = block_masks[b_idx]
                    while m_b:
                        lsb = m_b & -m_b
                        bit_idx = lsb.bit_length() - 1
                        if counts[bit_idx] == 1:
                            uncovered_if_removed |= (1 << bit_idx)
                        m_b ^= lsb
                        
                    if not uncovered_if_removed:
                        # Redundant, handled by remove_redundants, but we can remove here too
                        selected.remove(b_idx)
                        improved = True
                        break # Restart loop as counts changed
                        
                    # Find a block c not in sel_set that covers most of uncovered_if_removed
                    best_c = -1
                    max_cov = 0
                    for c_idx in range(n_k_blocks):
                        if c_idx in sel_set:
                            continue
                        m_c = block_masks[c_idx] & uncovered_if_removed
                        cov = m_c.bit_count()
                        if cov > max_cov:
                            max_cov = cov
                            best_c = c_idx
                            
                    if best_c != -1 and max_cov == popcount(uncovered_if_removed):
                        # If we swap b_idx with best_c, all previously uncovered (by b_idx removal) are covered.
                        # Is the new solution valid?
                        # Yes, because:
                        # 1. Subsets covered by others in S remain covered.
                        # 2. Subsets covered only by b_idx are now covered by best_c.
                        # So S' = (S \ {b_idx}) U {best_c} is valid.
                        # Size is same.
                        # Does it help? Maybe it allows other blocks to become redundant.
                        
                        # Perform swap
                        selected.remove(b_idx)
                        selected.append(best_c)
                        sel_set.remove(b_idx)
                        sel_set.add(best_c)
                        improved = True
                        break # Restart loop
                        
            return selected
            
        # Run swap improvement
        selected = try_swap_improvement(selected)
        
        # Final redundancy removal
        selected = remove_redundants(selected)
        
        return [list(k_blocks[i]) for i in selected]
    