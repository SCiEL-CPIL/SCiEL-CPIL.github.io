import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # 2. Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # 3. Build dense coverage matrix M (num_blocks x num_t)
    # M[i, j] = 1 if block i covers t-subset j
    # Using uint8 to save memory
    M = np.zeros((num_blocks, num_t), dtype=np.uint8)
    
    for i, block in enumerate(all_blocks):
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            M[i, idx] = 1

    # 4. Greedy Set Cover
    uncovered = np.ones(num_t, dtype=np.uint8)
    selected_indices = []
    
    while uncovered.any():
        if time.time() > deadline - 10.0:
            break
            
        # Calculate gain for each block: number of uncovered t-subsets it covers
        # M is (num_blocks, num_t), uncovered is (num_t,)
        # Result is (num_blocks,)
        gains = M @ uncovered
        
        # Find block with max gain
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
        
        selected_indices.append(best_idx)
        
        # Update uncovered: set to 0 where the chosen block covers
        M_row = M[best_idx]
        uncovered[M_row == 1] = 0

    # 5. Redundancy Removal
    if not selected_indices:
        return []
        
    selected_indices = np.array(selected_indices, dtype=np.int64)
    
    # Compute coverage counts
    M_selected = M[selected_indices]
    coverage_counts = M_selected.sum(axis=0)
    
    keep_mask = np.ones(len(selected_indices), dtype=bool)
    
    for i in range(len(selected_indices)):
        if not keep_mask[i]:
            continue
            
        if time.time() > deadline - 5.0:
            break
            
        block_row = M[selected_indices[i]]
        counts_of_covered = coverage_counts[block_row == 1]
        
        if len(counts_of_covered) > 0 and np.min(counts_of_covered) == 1:
            # Essential
            continue
        else:
            # Redundant
            keep_mask[i] = False
            coverage_counts -= block_row
            
    final_indices = selected_indices[keep_mask]
    final_blocks = [list(all_blocks[i]) for i in final_indices]
    
    return final_blocks
