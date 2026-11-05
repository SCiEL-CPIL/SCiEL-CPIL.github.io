import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Randomized Greedy and Optimized Local Search.
    """
    # 1. Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    c_k_t = int(len(list(itertools.combinations(range(k), t))))
    block_t_indices = np.zeros((num_blocks, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        for j, ts in enumerate(itertools.combinations(block, t)):
             block_t_indices[i, j] = t_sub_to_idx[ts]
             
    # Map block tuple to its index in all_blocks for fast lookup
    block_tuple_to_idx = {b: i for i, b in enumerate(all_blocks)}

    def greedy_construct(rng):
        uncovered = np.ones(num_t_subs, dtype=np.int8)
        used_block_mask = np.zeros(num_blocks, dtype=bool)
        blocks_used = []
        
        while uncovered.sum() > 0:
            gains = uncovered[block_t_indices].sum(axis=1)
            gains[used_block_mask] = -1
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            blocks_used.append(all_blocks[best_idx])
            used_block_mask[best_idx] = True
            uncovered[block_t_indices[best_idx]] = 0
            
        return [list(b) for b in blocks_used]

    def local_search_deletion(blocks):
        # Initialize counts
        t_sub_count = np.zeros(num_t_subs, dtype=np.int32)
        
        # Store t-indices for the blocks currently in the solution
        solution_t_indices = []
        for b in blocks:
            idx = block_tuple_to_idx[tuple(b)]
            t_indices = block_t_indices[idx]
            solution_t_indices.append(t_indices)
            t_sub_count[t_indices] += 1
            
        # Single backward pass is sufficient for minimal cover
        for i in range(len(blocks) - 1, -1, -1):
            t_indices = solution_t_indices[i]
            if np.min(t_sub_count[t_indices]) > 1:
                t_sub_count[t_indices] -= 1
                blocks.pop(i)
                solution_t_indices.pop(i)
                
        return blocks

    # Strategy: Run a few randomized greedy starts, keep the best after deletion
    best_blocks = []
    best_len = float('inf')
    start_time = time.time()
    
    # Deterministic start (seed 0)
    rng = random.Random(0)
    blocks = greedy_construct(rng)
    blocks = local_search_deletion(blocks)
    best_len = len(blocks)
    best_blocks = blocks.copy()
    
    # Random restarts
    for seed in range(1, 20):
        if time.time() - start_time > 100:
            break
        rng = random.Random(seed)
        blocks = greedy_construct(rng)
        blocks = local_search_deletion(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks.copy()
            
    return best_blocks
