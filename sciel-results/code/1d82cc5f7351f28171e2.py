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
    
    # Precompute bitsets for each block
    block_bitsets = []
    block_subset_indices = []
    for b in all_blocks:
        subs = itertools.combinations(b, t)
        indices = [subset_to_idx[s] for s in subs]
        block_subset_indices.append(indices)
        
        bits = 0
        for idx in indices:
            bits |= (1 << idx)
        block_bitsets.append(bits)
        
    # Greedy Initialization
    mask_all = (1 << total_subsets) - 1
    uncovered_bits = mask_all
    current_blocks = []
    used_blocks = [False] * num_blocks
    
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
        
    # ILS (Iterated Local Search)
    start_time = time.time()
    time_limit = 115
    
    best_cost = len(current_blocks)
    best_blocks = list(current_blocks)
    
    # Lower bound from task description
    lower_bound = 264
    
    # Helper to update counts
    def update_counts(blocks, counts, add=True):
        for bi in blocks:
            for idx in block_subset_indices[bi]:
                if add:
                    counts[idx] += 1
                else:
                    counts[idx] -= 1

    # Local Search Function
    def local_search(blocks):
        counts = np.zeros(total_subsets, dtype=np.int32)
        update_counts(blocks, counts, True)
        
        current_set = set(blocks)
        changed = True
        while changed:
            changed = False
            # 1. Remove redundant blocks
            for i in range(len(blocks) - 1, -1, -1):
                bi = blocks[i]
                redundant = True
                for idx in block_subset_indices[bi]:
                    if counts[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    current_set.remove(bi)
                    for idx in block_subset_indices[bi]:
                        counts[idx] -= 1
                    changed = True
            
            # 2. Add blocks if uncovered
            uncovered_bits = 0
            for i in range(total_subsets):
                if counts[i] == 0:
                    uncovered_bits |= (1 << i)
            
            while uncovered_bits != 0:
                best_gain = 0
                best_block_idx = -1
                for bi in range(num_blocks):
                    if bi in current_set:
                        continue
                    gain = (uncovered_bits & block_bitsets[bi]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = bi
                if best_block_idx == -1:
                    break
                blocks.append(best_block_idx)
                current_set.add(best_block_idx)
                for idx in block_subset_indices[best_block_idx]:
                    counts[idx] += 1
                uncovered_bits &= ~block_bitsets[best_block_idx]
                changed = True
                
        return blocks, counts

    # Run ILS
    current_sol = list(best_blocks)
    
    while time.time() - start_time < time_limit:
        if best_cost <= lower_bound:
            break
            
        # Perturb
        if len(current_sol) > 0:
            # Remove 1-5 random blocks to allow escape from local optima
            # Too many removals makes it hard to recover
            num_remove = min(len(current_sol), random.randint(1, 5))
            indices_to_remove = random.sample(range(len(current_sol)), num_remove)
            indices_to_remove.sort(reverse=True)
            for i in indices_to_remove:
                current_sol.pop(i)
        
        # Local Search
        improved_sol, _ = local_search(current_sol)
        
        if len(improved_sol) < best_cost:
            best_cost = len(improved_sol)
            best_blocks = list(improved_sol)
            current_sol = list(improved_sol)
        else:
            # If no improvement, revert to best solution for next perturbation
            current_sol = list(best_blocks)
            
    # Final cleanup on best_blocks
    final_counts = np.zeros(total_subsets, dtype=np.int32)
    update_counts(best_blocks, final_counts, True)
    changed = True
    while changed:
        changed = False
        for i in range(len(best_blocks) - 1, -1, -1):
            bi = best_blocks[i]
            redundant = True
            for idx in block_subset_indices[bi]:
                if final_counts[idx] == 1:
                    redundant = False
                    break
            if redundant:
                best_blocks.pop(i)
                for idx in block_subset_indices[bi]:
                    final_counts[idx] -= 1
                changed = True

    result = [list(all_blocks[bi]) for bi in best_blocks]
    return result
