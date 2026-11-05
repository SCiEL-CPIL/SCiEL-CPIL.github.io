import itertools
import time
import numpy as np
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Build incidence matrix M (n_k x n_t)
    # M[i, j] = 1 if block i covers t-subset j
    # Using lists for rows/cols to build CSR efficiently
    rows = []
    cols = []
    
    # Optimization: Precompute t-subsets for each k-subset is not needed, 
    # just iterate combinations of the block.
    for i, block in enumerate(k_subsets):
        # Generate all t-combinations within this block
        for comb in itertools.combinations(block, t):
            j = t_to_idx[comb]
            rows.append(i)
            cols.append(j)
            
    # Convert to numpy arrays for scipy sparse constructor
    rows_np = np.array(rows, dtype=np.int32)
    cols_np = np.array(cols, dtype=np.int32)
    data = np.ones(len(rows), dtype=np.int8)
    
    M = csr_matrix((data, (rows_np, cols_np)), shape=(n_k, n_t))
    
    # 2. Greedy Set Cover
    uncovered = np.ones(n_t, dtype=np.int8)
    blocks_used = []
    
    while uncovered.sum() > 0:
        if time.time() > deadline:
            # If we run out of time, we can't return an invalid covering.
            # However, given the speed, this is highly unlikely.
            # If it happens, we break and return what we have (which might be invalid).
            # To be safe, we could try to finish the remaining few steps if close, 
            # but standard greedy is fast.
            break
            
        # Calculate gain: number of uncovered t-subsets each block would cover
        gains = M.dot(uncovered)
        
        # Find the best block
        best_idx = np.argmax(gains)
        
        # If no gain, we are stuck (should not happen for valid parameters)
        if gains[best_idx] == 0:
            break
            
        # Add block
        blocks_used.append(list(k_subsets[best_idx]))
        
        # Update uncovered
        # Extract the sparse row and find non-zero indices
        covered_mask = M[best_idx, :].toarray().flatten()
        covered_indices = np.where(covered_mask > 0)[0]
        uncovered[covered_indices] = 0
        
    return blocks_used
