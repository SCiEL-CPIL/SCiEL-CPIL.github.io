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
    block_t_list = []
    for block in k_blocks:
        covered = [t_index[ts] for ts in itertools.combinations(block, t)]
        block_t_list.append(covered)
    
    # Precompute full coverage
    full_coverage = np.zeros(n_t, dtype=np.int32)
    for covered in block_t_list:
        for idx in covered:
            full_coverage[idx] += 1
            
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple trials to find a better minimal cover
    for trial in range(10):
        random.seed(42 + trial)
        order = list(range(n_k))
        random.shuffle(order)
        
        active = np.ones(n_k, dtype=bool)
        coverage = full_coverage.copy()
        
        for i in order:
            # Check if block i is essential (covers any subset with count 1)
            essential = False
            for idx in block_t_list[i]:
                if coverage[idx] == 1:
                    essential = True
                    break
            
            if not essential:
                # Remove block i
                active[i] = False
                for idx in block_t_list[i]:
                    coverage[idx] -= 1
                    
        count = np.sum(active)
        if count < best_count:
            best_count = count
            best_blocks = [list(k_blocks[i]) for i in range(n_k) if active[i]]
            
    return best_blocks
