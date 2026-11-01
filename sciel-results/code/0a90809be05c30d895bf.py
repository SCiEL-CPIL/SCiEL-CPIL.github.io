import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    subset_to_id = {s: i for i, s in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(k_subsets)
    block_to_idx = {s: i for i, s in enumerate(k_subsets)}
    
    # Precompute covered subset indices for each block
    block_covered = []
    for block in k_subsets:
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(subset_to_id[ts])
        block_covered.append(covered)
        
    # Build dense boolean matrix for Greedy
    # M[i, j] = 1 if block i covers subset j
    M = np.zeros((num_k, num_t), dtype=np.int8)
    for i, covered in enumerate(block_covered):
        if covered:
            M[i, covered] = 1
            
    # 2. Greedy Construction
    uncovered = np.ones(num_t, dtype=np.int8)
    selected_blocks = []
    
    while uncovered.sum() > 0:
        if time.time() > deadline:
            break
            
        scores = M.dot(uncovered)
        best_idx = np.argmax(scores)
        
        if scores[best_idx] == 0:
            break
            
        selected_blocks.append(best_idx)
        uncovered[M[best_idx] == 1] = 0
        
    # 3. Local Search
    current_blocks = set(k_subsets[i] for i in selected_blocks)
    
    coverage_counts = np.zeros(num_t, dtype=np.int16)
    for block in current_blocks:
        idx = block_to_idx[block]
        covered = block_covered[idx]
        if covered:
            coverage_counts[covered] += 1
            
    def try_remove_redundant():
        nonlocal current_blocks, coverage_counts
        removed = True
        while removed:
            removed = False
            for block in list(current_blocks):
                if time.time() > deadline:
                    return
                idx = block_to_idx[block]
                covered = block_covered[idx]
                if covered:
                    if np.min(coverage_counts[covered]) > 1:
                        current_blocks.remove(block)
                        coverage_counts[covered] -= 1
                        removed = True
                        
    def try_swap_improve():
        nonlocal current_blocks, coverage_counts
        improved = False
        for block in list(current_blocks):
            if time.time() > deadline:
                return
            
            idx_old = block_to_idx[block]
            covered_old = block_covered[idx_old]
            
            # Identify unique subsets
            # unique_mask = coverage_counts[covered_old] == 1
            # unique_indices = covered_old[unique_mask]
            
            block_set = set(block)
            for p in block:
                for q in range(v):
                    if q in block_set:
                        continue
                    new_block_tuple = tuple(sorted((block_set - {p}) | {q}))
                    if new_block_tuple in current_blocks:
                        continue
                    
                    idx_new = block_to_idx[new_block_tuple]
                    covered_new = block_covered[idx_new]
                    
                    # Check if new block covers all unique subsets of old block
                    is_valid = True
                    for s in covered_old:
                        if coverage_counts[s] == 1:
                            if M[idx_new, s] == 0:
                                is_valid = False
                                break
                                
                    if is_valid:
                        current_blocks.remove(block)
                        coverage_counts[covered_old] -= 1
                        
                        current_blocks.add(new_block_tuple)
                        coverage_counts[covered_new] += 1
                        
                        improved = True
                        break
            if improved:
                break
                
    try_remove_redundant()
    
    while time.time() < deadline - 2.0:
        try_swap_improve()
        try_remove_redundant()
        
    return [list(b) for b in current_blocks]
