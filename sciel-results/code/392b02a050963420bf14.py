"""Covering C(19,8,5): randomized minimal cover via sequential removal of non-essential blocks."""
import numpy as np
import itertools
import random

def construct(v, k, t):
    # Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(k_blocks)
    
    # For each block, compute the list of t-subset indices it covers
    # Store as a list of numpy arrays for vectorized operations
    block_t_list = []
    for block in k_blocks:
        covered = np.array([t_index[ts] for ts in itertools.combinations(block, t)], dtype=np.int32)
        block_t_list.append(covered)
    
    # Precompute full coverage
    # Using a sparse-like approach or just summing vectors
    # Since n_t is ~11k and n_k is ~75k, a dense matrix is 75k*11k*4 bytes ~ 3.5GB which is too much.
    # So we stick to the list of indices approach but vectorize the check.
    
    full_coverage = np.zeros(n_t, dtype=np.int32)
    for covered in block_t_list:
        full_coverage[covered] += 1
            
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple trials to find a better minimal cover
    # Increase trials as vectorized ops are faster
    for trial in range(30):
        random.seed(42 + trial)
        order = list(range(n_k))
        random.shuffle(order)
        
        active = np.ones(n_k, dtype=bool)
        coverage = full_coverage.copy()
        
        for i in order:
            # Check if block i is essential (covers any subset with count 1)
            # Vectorized check: are any of the covered indices having count 1?
            if np.any(coverage[block_t_list[i]] == 1):
                continue
            
            # Remove block i
            active[i] = False
            coverage[block_t_list[i]] -= 1
                    
        count = np.sum(active)
        if count < best_count:
            best_count = count
            best_blocks = [list(k_blocks[i]) for i in range(n_k) if active[i]]
            
    return best_blocks
