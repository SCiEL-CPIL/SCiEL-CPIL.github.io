import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Precompute coverage mask for each block
    block_masks_bool = np.zeros((n_k_blocks, n_t_subs), dtype=bool)
    for i, block in enumerate(all_k_blocks):
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            block_masks_bool[i, idx] = True
            
    block_masks_int = block_masks_bool.astype(np.uint8)
    
    # 2. Search Loop
    best_blocks = None
    best_count = float('inf')
    
    while True:
        if time.time() > deadline:
            break
            
        covered = np.zeros(n_t_subs, dtype=np.uint8)
        num_covered = 0
        current_blocks = []
        
        while num_covered < n_t_subs:
            if time.time() > deadline:
                break
                
            uncovered = 1 - covered  # uint8 array
            
            # Calculate gains
            gains = block_masks_int.dot(uncovered)
            
            positive_idx = np.where(gains > 0)[0]
            
            if len(positive_idx) == 0:
                break
            
            K = 10  # Top 10 candidates
            if len(positive_idx) <= K:
                candidates = positive_idx
            else:
                pos_gains = gains[positive_idx]
                part_idx = np.argpartition(pos_gains, -K)[-K:]
                candidates = positive_idx[part_idx]
                
            chosen_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            current_blocks.append(all_k_blocks[chosen_idx])
            
            num_covered += gains[chosen_idx]
            covered = np.maximum(covered, block_masks_int[chosen_idx])
            
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
    if best_blocks is None:
        return []
        
    return [list(b) for b in best_blocks]
