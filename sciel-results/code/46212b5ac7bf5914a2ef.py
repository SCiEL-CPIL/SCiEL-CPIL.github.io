import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    points = list(range(v))
    
    # Map each t-subset to an index
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute masks for all possible blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # block_masks[i] is an integer where bit j is 1 if block i covers t_subset j
    block_masks = [0] * n_blocks
    
    # Precomputation cost: O(n_blocks * C(k,t))
    for i, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # Greedy Set Cover
    uncovered_mask = (1 << n_t) - 1
    selected_blocks = []
    
    while uncovered_mask:
        if time.time() > deadline:
            break
            
        best_block_idx = -1
        best_gain = 0
        
        for i in range(n_blocks):
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        if best_block_idx == -1:
            break
            
        selected_blocks.append(best_block_idx)
        uncovered_mask &= ~block_masks[best_block_idx]
        
    # Local Search: Remove redundant blocks
    counts = [0] * n_t
    for b_idx in selected_blocks:
        mask = block_masks[b_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
            
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_selected = []
        for b_idx in selected_blocks:
            mask = block_masks[b_idx]
            redundant = True
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] <= 1:
                    redundant = False
                    break
                m ^= lsb
                
            if redundant:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                changed = True
            else:
                new_selected.append(b_idx)
                
        selected_blocks = new_selected
    
    # Swap-based local search: try to replace blocks with alternatives
    # that enable further redundancy removal
    if time.time() < deadline - 20:
        random.seed(42)
        improved = True
        max_swap_iterations = 50
        swap_iter = 0
        
        while improved and swap_iter < max_swap_iterations and time.time() < deadline - 10:
            improved = False
            swap_iter += 1
            
            # Recompute counts
            counts = [0] * n_t
            for b_idx in selected_blocks:
                mask = block_masks[b_idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lsb
            
            # Try swapping each selected block with an alternative
            for pos, b_idx in enumerate(selected_blocks):
                if time.time() > deadline - 10:
                    break
                
                old_mask = block_masks[b_idx]
                
                # Find t-subsets uniquely covered by this block (count == 1)
                unique_covered = []
                m = old_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        unique_covered.append(idx)
                    m ^= lsb
                
                if not unique_covered:
                    # This block is redundant - skip (should have been removed)
                    continue
                
                # Try to find a replacement block that covers all uniquely covered t-subsets
                # and has high overlap with other selected blocks
                best_replacement = -1
                best_score = -1
                
                # Sample some candidate blocks for efficiency
                candidates = random.sample(range(n_blocks), min(500, n_blocks))
                
                for cand in candidates:
                    if cand == b_idx:
                        continue
                    cand_mask = block_masks[cand]
                    
                    # Check if candidate covers all uniquely covered t-subsets
                    covers_unique = True
                    for idx in unique_covered:
                        if not (cand_mask >> idx) & 1:
                            covers_unique = False
                            break
                    
                    if not covers_unique:
                        continue
                    
                    # Score: how many t-subsets does the candidate share with other selected blocks?
                    # Higher overlap means less new coverage needed, potentially enabling more removals
                    overlap_score = 0
                    for other_idx in selected_blocks:
                        if other_idx == b_idx:
                            continue
                        overlap_score += (cand_mask & block_masks[other_idx]).bit_count()
                    
                    if overlap_score > best_score:
                        best_score = overlap_score
                        best_replacement = cand
                
                if best_replacement != -1:
                    # Perform the swap
                    selected_blocks[pos] = best_replacement
                    improved = True
                    break  # Restart the outer loop after a swap
        
        # Final redundancy removal after swaps
        counts = [0] * n_t
        for b_idx in selected_blocks:
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_selected = []
            for b_idx in selected_blocks:
                mask = block_masks[b_idx]
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                    
                if redundant:
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_selected.append(b_idx)
                    
            selected_blocks = new_selected
    
    # Convert indices back to blocks
    return [list(all_blocks[i]) for i in selected_blocks]
