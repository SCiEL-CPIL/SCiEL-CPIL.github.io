import itertools
import sys
import random

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute block_masks for greedy and later use
    # And blocks_covering_subset for local search
    block_masks = [0] * n_k_blocks
    blocks_covering_subset = [[] for _ in range(n_t_subs)]
    
    for i, block in enumerate(k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
            blocks_covering_subset[idx].append(i)
        block_masks[i] = mask
        
    # 3. Greedy Set Cover
    uncovered = (1 << n_t_subs) - 1
    selected_indices = []
    
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
            
        selected_indices.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    # 4. Local Search: Remove and Swap
    # Initialize counts
    counts = [0] * n_t_subs
    selected_set = set(selected_indices)
    
    for idx in selected_indices:
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            m ^= lsb
            
    def remove_block(idx):
        nonlocal counts
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] -= 1
            m ^= lsb
        selected_set.remove(idx)
        
    def add_block(idx):
        nonlocal counts
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            m ^= lsb
        selected_set.add(idx)
        
    # Local Search Loop
    # We iterate until no improvement (removal or swap) is found.
    # To avoid infinite loops in swaps, we can limit iterations or use a "best so far" tracker, 
    # but typically removals reduce size, and swaps enable removals.
    # Swaps alone don't reduce size, but they change the configuration.
    # We should prioritize removals. If no removals, do swaps. If no swaps, stop.
    
    max_iterations = 100 # Safety valve
    for _ in range(max_iterations):
        improved = False
        
        # Phase 1: Remove redundant blocks
        # We need to re-check all blocks because counts change.
        # Optimization: Only check blocks that might be redundant? 
        # A block is redundant if it has no unique t-subsets.
        # We can iterate through selected_set.
        
        to_remove = []
        for idx in list(selected_set):
            m = block_masks[idx]
            has_unique = False
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    has_unique = True
                    break
                m ^= lsb
            if not has_unique:
                to_remove.append(idx)
                
        if to_remove:
            for idx in to_remove:
                remove_block(idx)
            improved = True
            # After removals, we can immediately try to remove more or swap.
            # Continue to next iteration of the main loop to re-evaluate everything.
            continue 
            
        # Phase 2: Swap
        # Try to find a swap that might enable a removal later.
        # For each block in selected_set, check if we can swap it with another block.
        
        swap_found = False
        for idx_out in list(selected_set):
            # Find unique t-subsets for idx_out
            unique_subs = []
            m = block_masks[idx_out]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    unique_subs.append(bit_idx)
                m ^= lsb
                
            if not unique_subs:
                # Should have been removed in Phase 1
                continue
                
            # Find candidate blocks that cover all unique_subs
            # Start with the list of blocks covering the first unique sub
            candidates = set(blocks_covering_subset[unique_subs[0]])
            
            for sub in unique_subs[1:]:
                # Intersect with blocks covering this sub
                # Optimization: if candidates becomes empty, break
                if not candidates:
                    break
                candidates.intersection_update(blocks_covering_subset[sub])
                
            # Remove blocks already in selected_set
            candidates -= selected_set
            
            if candidates:
                # Pick a candidate. 
                # Heuristic: Pick the one that covers the most "low count" subsets?
                # Or just pick the first one to keep it simple and fast.
                # Let's pick the one with the highest coverage of currently low-count subsets (count == 1 or 2) to potentially create new redundancies.
                # Actually, any valid swap is good. Let's pick the first one for speed.
                # To be slightly smarter, we can prefer blocks that cover many subsets with count==1.
                
                best_in = -1
                best_score = -1
                for idx_in in candidates:
                    m_in = block_masks[idx_in]
                    score = 0
                    # Count how many subsets it covers that are currently "tight" (count <= 2)
                    # This is expensive. Let's just pick the first candidate.
                    best_in = idx_in
                    break
                    
                if best_in != -1:
                    # Perform swap
                    remove_block(idx_out)
                    add_block(best_in)
                    swap_found = True
                    improved = True
                    break # Restart search after a swap
                    
        if not improved:
            break
            
    return [list(k_blocks[i]) for i in selected_set]
