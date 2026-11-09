import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 2. Map t-subsets to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subs = len(t_subsets)
    
    # Number of 64-bit words needed
    num_words = (num_t_subs + 63) // 64
    
    # Precompute bitsets for all blocks
    bitsets = np.zeros((num_blocks, num_words), dtype=np.uint64)
    
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            idx = t_to_idx[ts]
            word_idx = idx // 64
            bit_idx = idx % 64
            bitsets[i, word_idx] |= (1 << bit_idx)
            
    best_blocks = None
    best_len = float('inf')
    
    num_runs = 20 # More runs to explore tie-breaking variations
    
    for run in range(num_runs):
        if time.time() > deadline:
            break
            
        uncovered = np.ones(num_words, dtype=np.uint64)
        valid_bits = num_t_subs % 64
        if valid_bits == 0:
            valid_bits = 64
        if valid_bits < 64:
            uncovered[-1] = (1 << valid_bits) - 1
            
        used_blocks = np.zeros(num_blocks, dtype=bool)
        current_blocks = []
        
        # Allocate buffer locally to avoid state issues
        buffer = np.zeros((num_blocks, num_words), dtype=np.uint64)
        
        while True:
            if time.time() > deadline:
                break
                
            np.bitwise_and(bitsets, uncovered[None, :], out=buffer)
            gains = buffer.sum(axis=1)
            gains[used_blocks] = 0
            
            max_gain = np.max(gains)
            
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) > 1:
                chosen_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                chosen_idx = candidates[0]
                
            used_blocks[chosen_idx] = True
            current_blocks.append(all_blocks[chosen_idx])
            
            uncovered = np.bitwise_and(uncovered, ~bitsets[chosen_idx])
            
            if not np.any(uncovered):
                break
                
        if len(current_blocks) < best_len and np.all(uncovered == 0):
            best_len = len(current_blocks)
            best_blocks = current_blocks
            
    if best_blocks is None:
        # Fallback: Return a deterministic greedy result if no run completed (unlikely)
        # Or just return the last run's result if it was valid?
        # If best_blocks is None, it means no run completed successfully.
        # This shouldn't happen.
        return []
        
    return best_blocks
