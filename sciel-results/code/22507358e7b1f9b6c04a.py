import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Precompute bitmasks for each block
    # mask[i] is a bitmask of t-subsets covered by all_k_blocks[i]
    block_masks = [0] * n_k_blocks
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    full_mask = (1 << n_t_subsets) - 1

    def greedy_cover(seed=0):
        random.seed(seed)
        uncovered_mask = full_mask
        selected_blocks = []
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            best_idx = -1
            best_gain = 0
            
            for i in range(n_k_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                if best_gain == 56: 
                    break
            
            if best_idx == -1:
                break
                
            selected_blocks.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            
        return selected_blocks

    def local_search(blocks_indices, seed=0):
        random.seed(seed)
        blocks_indices = list(blocks_indices)
        
        # Build counts
        counts = [0] * n_t_subsets
        for b_idx in blocks_indices:
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Pass 1: Remove redundant blocks
            to_remove = []
            for b_idx in blocks_indices:
                mask = block_masks[b_idx]
                m = mask
                essential = False
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        essential = True
                        break
                    m ^= lsb
                if not essential:
                    to_remove.append(b_idx)
            
            if to_remove:
                remove_set = set(to_remove)
                blocks_indices = [b for b in blocks_indices if b not in remove_set]
                for b_idx in to_remove:
                    mask = block_masks[b_idx]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                improved = True
                continue
            
            # Pass 2: 1-for-1 Swap
            swap_found = False
            for i, b_idx in enumerate(blocks_indices):
                if time.time() > deadline:
                    break
                
                mask_b = block_masks[b_idx]
                critical_mask = 0
                m = mask_b
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        critical_mask |= lsb
                    m ^= lsb
                
                if critical_mask == 0:
                    blocks_indices.remove(b_idx)
                    m = mask_b
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    improved = True
                    swap_found = True
                    break
            
            if improved:
                continue
                
            # If no simple removal, try swap
            # Check if critical_mask can be covered by another block
            # We limit the work per swap attempt to avoid timeouts
            for i, b_idx in enumerate(blocks_indices):
                if time.time() > deadline:
                    break
                
                mask_b = block_masks[b_idx]
                critical_mask = 0
                m = mask_b
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        critical_mask |= lsb
                    m ^= lsb
                
                if critical_mask == 0:
                    continue
                
                if critical_mask.bit_count() > 56:
                    continue
                
                found_replacement = False
                for j in range(n_k_blocks):
                    if (critical_mask & block_masks[j]) == critical_mask:
                        if j in blocks_indices:
                             blocks_indices.remove(b_idx)
                             m = mask_b
                             while m:
                                 lsb = m & -m
                                 idx = lsb.bit_length() - 1
                                 counts[idx] -= 1
                                 m ^= lsb
                        else:
                             blocks_indices[i] = j
                             m = mask_b
                             while m:
                                 lsb = m & -m
                                 idx = lsb.bit_length() - 1
                                 counts[idx] -= 1
                                 m ^= lsb
                             m = block_masks[j]
                             while m:
                                 lsb = m & -m
                                 idx = lsb.bit_length() - 1
                                 counts[idx] += 1
                                 m ^= lsb
                             
                        improved = True
                        found_replacement = True
                        break
                
                if found_replacement:
                    break
            
        return blocks_indices

    # Run Greedy
    initial_blocks = greedy_cover(42)
    
    # Run Local Search
    current_blocks = local_search(initial_blocks, 42)
    
    best_blocks = current_blocks
    best_count = len(current_blocks)
    
    if time.time() < deadline - 10:
        for seed in range(1, 5):
            if time.time() > deadline - 5:
                break
            
            perturbed = list(initial_blocks)
            if len(perturbed) > 5:
                to_remove = random.sample(perturbed, 5)
                remove_set = set(to_remove)
                perturbed = [b for b in perturbed if b not in remove_set]
                
                covered_mask = 0
                for b_idx in perturbed:
                    covered_mask |= block_masks[b_idx]
                holes_mask = full_mask & ~covered_mask
                
                new_blocks = []
                uncovered = holes_mask
                while uncovered:
                    if time.time() > deadline - 2:
                        break
                    best_idx = -1
                    best_gain = 0
                    for i in range(n_k_blocks):
                        gain = (uncovered & block_masks[i]).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = i
                    if best_idx == -1:
                        break
                    new_blocks.append(best_idx)
                    uncovered &= ~block_masks[best_idx]
                
                candidate = perturbed + new_blocks
                candidate = local_search(candidate, seed)
                
                if len(candidate) < best_count:
                    best_blocks = candidate
                    best_count = len(candidate)
                    initial_blocks = list(candidate)

    final_blocks = [list(all_k_blocks[i]) for i in best_blocks]
    return final_blocks
