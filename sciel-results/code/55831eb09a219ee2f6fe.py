import itertools
import time
import numpy as np
from scipy import sparse

def construct(v, k, t):
    start_time = time.time()
    # Allow 140s for safety, leaving 10s for cleanup/return
    deadline = 140.0 
    
    # 1. Generate all blocks and t-subsets
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    all_t_subsets = list(itertools.combinations(points, t))
    
    # Map t-subsets to indices for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    N_t = len(all_t_subsets)
    N_b = len(all_blocks)
    
    # 2. Build Sparse Incidence Matrix M (N_b x N_t)
    # M[i, j] = 1 if t-subset j is in block i
    rows = []
    cols = []
    
    # Pre-compute t-subsets for each block to avoid repeated itertools calls?
    # itertools.combinations is fast, but let's see.
    # 75582 blocks * 56 combinations = 4.2M calls.
    # This might take a few seconds.
    
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            j = t_subset_to_idx[ts]
            rows.append(i)
            cols.append(j)
            
    # Convert to CSR for efficient row slicing and matrix-vector product
    M = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N_b, N_t))
    
    # 3. Greedy Construction
    uncovered = np.ones(N_t, dtype=np.float32)
    selected_indices = []
    
    while True:
        if time.time() - start_time > deadline:
            break
            
        # Calculate gain for each block: number of currently uncovered t-subsets it covers
        gains = M @ uncovered
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
            
        selected_indices.append(best_idx)
        
        # Update uncovered mask
        # Get the column indices for the selected block
        start = M.indptr[best_idx]
        end = M.indptr[best_idx + 1]
        cols_to_update = M.indices[start:end]
        uncovered[cols_to_update] = 0.0
        
    # 4. Redundancy Removal
    # Calculate coverage counts for the selected blocks
    coverage_count = np.zeros(N_t, dtype=np.int32)
    for idx in selected_indices:
        start = M.indptr[idx]
        end = M.indptr[idx + 1]
        cols = M.indices[start:end]
        coverage_count[cols] += 1
        
    # Iteratively remove redundant blocks
    # A block is redundant if all its t-subsets are covered by at least one other selected block
    # i.e., if min(coverage_count[cols_in_block]) > 0
    changed = True
    while changed:
        changed = False
        # Iterate in reverse to handle index shifts easily if we didn't break, 
        # but we break on change, so order doesn't strictly matter for correctness, 
        # just for which block gets removed.
        for i in range(len(selected_indices) - 1, -1, -1):
            idx = selected_indices[i]
            start = M.indptr[idx]
            end = M.indptr[idx + 1]
            cols = M.indices[start:end]
            
            if np.min(coverage_count[cols]) > 0:
                # Block is redundant
                coverage_count[cols] -= 1
                selected_indices.pop(i)
                changed = True
                break # Restart the while loop to find new redundancies
                
    # 5. Return the blocks as lists of integers
    result = [list(all_blocks[idx]) for idx in selected_indices]
    
    return result
