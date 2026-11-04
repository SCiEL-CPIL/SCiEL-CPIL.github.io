import itertools
import time
import numpy as np
import scipy.sparse

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Generate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # 2. Generate all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_blocks)

    # 3. Build Sparse Matrix M (n_blocks x n_t)
    row_indices = []
    col_indices = []
    data = []
    
    # Optimization: Pre-compute combinations for each block
    for b_idx, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            s_idx = t_sub_to_idx.get(t_sub)
            if s_idx is not None:
                row_indices.append(b_idx)
                col_indices.append(s_idx)
                data.append(1)
                
    M = scipy.sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_t), dtype=np.int32)
    
    # 4. Greedy Construction
    uncovered = np.ones(n_t, dtype=np.int32)
    active_blocks = []
    
    while uncovered.sum() > 0:
        if time.time() > deadline - 15.0:
            break
            
        # Calculate gains: how many uncovered subsets each block covers
        gains = M @ uncovered
        
        # Pick best block
        best_b = np.argmax(gains)
        if gains[best_b] == 0:
            break # Should not happen
            
        active_blocks.append(best_b)
        
        # Update uncovered
        start = M.indptr[best_b]
        end = M.indptr[best_b + 1]
        cols = M.indices[start:end]
        uncovered[cols] = 0
        
    # 5. Redundancy Removal
    active_arr = np.array(active_blocks, dtype=np.int32)
    
    while time.time() < deadline - 5.0:
        if len(active_arr) == 0:
            break
            
        # Compute coverage counts
        # M[active_arr, :] is a sparse matrix. 
        # Multiplying by ones vector gives sum of each row? No, sum of each column?
        # We want counts for each t-subset (column).
        # counts[s] = sum of M[b, s] for b in active_arr
        # This is (M[active_arr, :].T @ ones(len(active_arr)))?
        # Or simply M[active_arr, :] @ np.ones(len(active_arr))? No.
        # M is (n_blocks, n_t).
        # We want column sums of the submatrix.
        # sub_M = M[active_arr, :]
        # counts = sub_M.sum(axis=0) -> returns a matrix, need to flatten.
        # scipy sparse sum axis=0 returns a matrix.
        
        sub_M = M[active_arr, :]
        counts = np.asarray(sub_M.sum(axis=0)).flatten()
        
        # Check for redundant blocks
        remove_mask = np.ones(len(active_arr), dtype=bool)
        
        for i in range(len(active_arr)):
            b_idx = active_arr[i]
            start = M.indptr[b_idx]
            end = M.indptr[b_idx + 1]
            if start == end:
                continue 
            
            cols = M.indices[start:end]
            min_count = np.min(counts[cols])
            
            if min_count > 1:
                remove_mask[i] = False # Keep
            else:
                remove_mask[i] = True # Remove
        
        if remove_mask.all():
            break # No blocks to remove
            
        active_arr = active_arr[~remove_mask]
        
    # 6. Construct Result
    result = [list(all_k_blocks[b]) for b in active_arr]
    return result
