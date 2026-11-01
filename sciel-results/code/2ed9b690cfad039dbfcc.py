import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    block_to_index = {b: i for i, b in enumerate(all_k_blocks)}
    
    block_cover = []
    for block in all_k_blocks:
        covered = np.array([t_subset_to_index[sub] for sub in itertools.combinations(block, t)], dtype=np.int32)
        block_cover.append(covered)
    
    best_solution = None
    best_count = float('inf')
    
    num_passes = 20 # Increase passes to find better solutions
    
    for p in range(num_passes):
        if time.time() > deadline - 30:
            break
            
        random.seed(p + 12345)
        
        uncovered = np.ones(num_t_subsets, dtype=bool)
        blocks_used = []
        
        while uncovered.any():
            if time.time() > deadline - 30:
                break
                
            gains = np.array([uncovered[covered].sum() for covered in block_cover])
            max_gain = gains.max()
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_block = random.choice(candidates)
            
            blocks_used.append(all_k_blocks[best_block])
            uncovered[block_cover[best_block]] = False
            
        block_indices = [block_to_index[b] for b in blocks_used]
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for idx in block_indices:
            cover_count[block_cover[idx]] += 1
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(block_indices) - 1, -1, -1):
                idx = block_indices[i]
                if cover_count[block_cover[idx]].min() > 1:
                    cover_count[block_cover[idx]] -= 1
                    block_indices.pop(i)
                    improved = True
                    if time.time() > deadline - 5:
                        break
        
        if len(block_indices) < best_count:
            best_count = len(block_indices)
            best_solution = [list(all_k_blocks[idx]) for idx in block_indices]
            
    if best_solution is None:
        return []
        
    return best_solution
