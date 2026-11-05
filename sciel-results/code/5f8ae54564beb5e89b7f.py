import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Generate all possible k-blocks and map them to their t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each t-subset to an index 0..C(v,t)-1
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute for each block, the indices of the t-subsets it covers
    k_choose_t = len(list(itertools.combinations(range(k), t)))
    block_subset_indices = np.empty((num_blocks, k_choose_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        for j, subset in enumerate(itertools.combinations(block, t)):
            block_subset_indices[i, j] = t_subset_to_idx[subset]
            
    # 2. Greedy Construction with Randomization
    def greedy_construct():
        uncovered = np.ones(num_t_subsets, dtype=bool)
        selected_blocks = []
        
        while uncovered.any():
            if time.time() > deadline:
                break
                
            # Calculate gain for all blocks
            gains = uncovered[block_subset_indices].sum(axis=1)
            
            # Add small random noise to break ties and escape local minima
            noise = np.random.uniform(0, 0.1, size=num_blocks)
            gains_noisy = gains + noise
            
            max_gain_idx = np.argmax(gains_noisy)
            max_gain = gains[max_gain_idx]
            
            if max_gain == 0:
                break
                
            selected_blocks.append(max_gain_idx)
            uncovered[block_subset_indices[max_gain_idx]] = False
            
        return selected_blocks
    
    # 3. Redundancy Removal
    def remove_redundant(selected_blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected_blocks:
            cover_count[block_subset_indices[b_idx]] += 1
            
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            new_selected = []
            for b_idx in selected_blocks:
                sub_indices = block_subset_indices[b_idx]
                if np.min(cover_count[sub_indices]) > 1:
                    cover_count[sub_indices] -= 1
                    changed = True
                else:
                    new_selected.append(b_idx)
                    
            selected_blocks = new_selected
            
        return selected_blocks
    
    # 4. Local Search: Try to replace blocks to reduce count
    def local_search(selected_blocks):
        if not selected_blocks:
            return selected_blocks
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected_blocks:
            cover_count[block_subset_indices[b_idx]] += 1
            
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # Try removing each block if possible
            for i in range(len(selected_blocks)):
                if time.time() > deadline:
                    break
                b_idx = selected_blocks[i]
                sub_indices = block_subset_indices[b_idx]
                
                # Check if removing this block leaves any subset uncovered
                if np.all(cover_count[sub_indices] > 1):
                    # Remove it
                    cover_count[sub_indices] -= 1
                    selected_blocks.pop(i)
                    improved = True
                    break  # Restart the loop after modification
        
        return selected_blocks
    
    # Run multiple randomized greedy constructions
    best_blocks = None
    num_attempts = 5
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
            
        # Different random seeds for each attempt
        np.random.seed(attempt * 12345 + 67890)
        
        selected = greedy_construct()
        if not selected:
            continue
            
        selected = remove_redundant(selected)
        selected = local_search(selected)
        
        if best_blocks is None or len(selected) < len(best_blocks):
            best_blocks = selected
            # Keep searching if we have time
            
    if best_blocks is None:
        # Fallback: single deterministic greedy
        np.random.seed(0)
        best_blocks = greedy_construct()
        best_blocks = remove_redundant(best_blocks)
        best_blocks = local_search(best_blocks)
        
    # Final redundancy removal
    best_blocks = remove_redundant(best_blocks)
    
    # 5. Convert indices back to lists of points
    result_blocks = [list(all_blocks[i]) for i in best_blocks]
    
    return result_blocks
