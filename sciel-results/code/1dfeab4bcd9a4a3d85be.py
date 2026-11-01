import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    total_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute list of subset indices for each block
    block_subset_indices = []
    for b in all_blocks:
        subs = itertools.combinations(b, t)
        indices = [subset_to_idx[s] for s in subs]
        block_subset_indices.append(indices)
        
    # Greedy Initialization using Bitsets
    mask_all = (1 << total_subsets) - 1
    uncovered_bits = mask_all
    current_blocks = []
    used_blocks = [False] * num_blocks
    
    # Precompute bitsets for each block
    block_bitsets = []
    for indices in block_subset_indices:
        bits = 0
        for idx in indices:
            bits |= (1 << idx)
        block_bitsets.append(bits)
        
    while uncovered_bits != 0:
        best_gain = 0
        best_block_idx = -1
        for bi in range(num_blocks):
            if used_blocks[bi]:
                continue
            gain = (uncovered_bits & block_bitsets[bi]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = bi
        
        if best_block_idx == -1:
            break
            
        current_blocks.append(best_block_idx)
        used_blocks[best_block_idx] = True
        uncovered_bits &= ~block_bitsets[best_block_idx]
        
    # Local Search
    counts = np.zeros(total_subsets, dtype=np.int32)
    for bi in current_blocks:
        for idx in block_subset_indices[bi]:
            counts[idx] += 1
            
    start_time = time.time()
    time_limit = 115
    T = 10.0
    cooling = 0.99
    
    best_cost = len(current_blocks)
    best_blocks = list(current_blocks)
    current_set = set(current_blocks)
    
    while time.time() - start_time < time_limit and T > 0.01:
        move_type = random.random()
        
        if move_type < 0.5 and current_blocks:
            # Try Remove
            idx_to_remove = random.randint(0, len(current_blocks) - 1)
            bi = current_blocks[idx_to_remove]
            
            redundant = True
            for s_idx in block_subset_indices[bi]:
                if counts[s_idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                current_blocks.pop(idx_to_remove)
                current_set.remove(bi)
                for s_idx in block_subset_indices[bi]:
                    counts[s_idx] -= 1
                if len(current_blocks) < best_cost:
                    best_cost = len(current_blocks)
                    best_blocks = list(current_blocks)
                    
        elif move_type < 0.9 and current_blocks:
            # Try Swap
            idx_to_swap = random.randint(0, len(current_blocks) - 1)
            old_bi = current_blocks[idx_to_swap]
            
            new_bi = random.randint(0, num_blocks - 1)
            if new_bi in current_set:
                continue
                
            # Check validity
            valid_swap = True
            new_bi_subs = set(block_subset_indices[new_bi])
            for s_idx in block_subset_indices[old_bi]:
                if counts[s_idx] == 1:
                    if s_idx not in new_bi_subs:
                        valid_swap = False
                        break
            
            if valid_swap:
                current_blocks[idx_to_swap] = new_bi
                current_set.remove(old_bi)
                current_set.add(new_bi)
                for s_idx in block_subset_indices[old_bi]:
                    counts[s_idx] -= 1
                for s_idx in block_subset_indices[new_bi]:
                    counts[s_idx] += 1
                    
        T *= cooling
        
    # Final Cleanup
    changed = True
    while changed:
        changed = False
        for i in range(len(current_blocks) - 1, -1, -1):
            bi = current_blocks[i]
            redundant = True
            for s_idx in block_subset_indices[bi]:
                if counts[s_idx] == 1:
                    redundant = False
                    break
            if redundant:
                current_blocks.pop(i)
                current_set.remove(bi)
                for s_idx in block_subset_indices[bi]:
                    counts[s_idx] -= 1
                changed = True
                
    # Use best found
    if len(best_blocks) < len(current_blocks):
        final_blocks = best_blocks
    else:
        final_blocks = current_blocks
        
    result = [list(all_blocks[bi]) for bi in final_blocks]
    return result
