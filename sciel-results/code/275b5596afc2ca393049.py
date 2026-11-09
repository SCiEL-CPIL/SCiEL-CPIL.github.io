import numpy as np
from itertools import combinations
import time

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    t_subs = list(combinations(range(v), t))
    t_sub_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Precompute all k-subsets
    k_subs = list(combinations(range(v), k))
    n_k = len(k_subs)
    
    # Build mapping from k-subset to its t-subset indices
    n_t_in_k = len(list(combinations(range(k), t)))
    
    block_t_indices = np.zeros((n_k, n_t_in_k), dtype=np.int32)
    for i, ks in enumerate(k_subs):
        for j, ts in enumerate(combinations(ks, t)):
            block_t_indices[i, j] = t_sub_idx[ts]
            
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    time_limit = 115.0
    
    while time.time() - start_time < time_limit:
        uncovered = np.ones(n_t, dtype=bool)
        n_uncovered = n_t
        blocks = []
        
        while n_uncovered > 0:
            gains = uncovered[block_t_indices].sum(axis=1)
            
            # Add random noise to break ties and explore different solution paths
            noisy_gains = gains + np.random.random(n_k)
            
            best_idx = np.argmax(noisy_gains)
            best_gain = gains[best_idx]
            
            if best_gain == 0:
                break
                
            blocks.append(k_subs[best_idx])
            
            block_indices = block_t_indices[best_idx]
            uncovered[block_indices] = False
            n_uncovered -= best_gain
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
    return best_blocks
