import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Precompute all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build coverage matrix A (num_blocks x num_t)
    # A[i, j] = 1 if block i covers t-subset j
    A = np.zeros((num_blocks, num_t), dtype=np.int8)
    
    # Precompute the list of t-subset indices for each block
    block_tsub_indices = []
    
    for i, block in enumerate(all_blocks):
        tsubs = list(itertools.combinations(block, t))
        indices = [t_sub_to_idx[sub] for sub in tsubs]
        block_tsub_indices.append(indices)
        A[i, indices] = 1
        
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy set cover instances
    max_runs = 500
    
    for run_id in range(max_runs):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(run_id * 7919 + 13)
        
        uncovered = np.ones(num_t, dtype=bool)
        selected_block_indices = []
        
        while uncovered.any():
            # Calculate gain for each block
            # Cast to int8 for efficient dot product
            gains = A.dot(uncovered.astype(np.int8))
            
            max_gain = np.max(gains)
            
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one candidate
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            selected_block_indices.append(best_idx)
            
            # Update uncovered
            uncovered &= ~A[best_idx].astype(bool)
            
        # Post-processing: Remove redundant blocks iteratively
        current_indices = list(selected_block_indices)
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 2.0:
                break
                
            # Calculate coverage counts
            cover_count = np.zeros(num_t, dtype=np.int32)
            for idx in current_indices:
                cover_count[block_tsub_indices[idx]] += 1
                
            new_indices = []
            for idx in current_indices:
                is_essential = False
                for sub_idx in block_tsub_indices[idx]:
                    if cover_count[sub_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_indices.append(idx)
                    
            if len(new_indices) < len(current_indices):
                current_indices = new_indices
                changed = True
                
        if len(current_indices) < best_count:
            best_count = len(current_indices)
            best_blocks = [list(all_blocks[i]) for i in current_indices]
            
    if best_blocks is None:
        return []
        
    return best_blocks
