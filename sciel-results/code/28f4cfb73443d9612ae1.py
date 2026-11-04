import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
import itertools
import time
import random

def construct(v, k, t):
    # Generate all possible blocks and t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map t-subsets to indices for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build sparse incidence matrix A
    # A[i, j] = 1 if t_subset i is contained in block j
    rows = []
    cols = []
    for j, block in enumerate(all_blocks):
        for t_sub in itertools.combinations(block, t):
            i = t_subset_to_idx[t_sub]
            rows.append(i)
            cols.append(j)
            
    A = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_t_subsets, n_blocks))
    
    # ILP Formulation
    c = np.ones(n_blocks, dtype=np.float64)
    constraints = LinearConstraint(A, 1, np.inf)
    bounds = Bounds(0, 1)
    integrality = np.ones(n_blocks, dtype=np.int8)
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    best_solution = None
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"time_limit": time_limit, "mip_rel_gap": 0.0})
        
        # Check if a solution was found
        if res.x is not None:
            x_int = np.round(res.x).astype(int)
            # Verify feasibility
            if np.all(A @ x_int >= 1):
                best_solution = [list(all_blocks[i]) for i in np.where(x_int == 1)[0]]
                
    except Exception:
        pass
        
    # Fallback: Simple Greedy Heuristic if ILP failed or timed out without solution
    if best_solution is None:
        uncovered = np.ones(n_t_subsets, dtype=bool)
        selected_blocks = []
        
        # Precompute coverage for each block for fast access
        # Using CSC for efficient column access (block -> subsets)
        A_csc = A.tocsc()
        
        while np.any(uncovered):
            # Calculate gain: number of uncovered subsets covered by each block
            # A_csc.T is CSR, but we can just use A_csc to get columns
            # Actually, A_csc.indices and A_csc.indptr give the rows (subset indices) for each column (block)
            # Gain for block j = sum(uncovered[A_csc.indices[A_csc.indptr[j]:A_csc.indptr[j+1]]])
            
            gains = np.zeros(n_blocks, dtype=np.int32)
            for j in range(n_blocks):
                start = A_csc.indptr[j]
                end = A_csc.indptr[j+1]
                if end > start:
                    gains[j] = np.sum(uncovered[A_csc.indices[start:end]])
                    
            best_block = int(np.argmax(gains))
            if gains[best_block] == 0:
                break # No block can cover more uncovered subsets
            
            selected_blocks.append(best_block)
            
            # Update uncovered
            start = A_csc.indptr[best_block]
            end = A_csc.indptr[best_block+1]
            uncovered[A_csc.indices[start:end]] = False
            
        # Post-process: Remove redundant blocks
        if selected_blocks:
            coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
            for b in selected_blocks:
                start = A_csc.indptr[b]
                end = A_csc.indptr[b+1]
                coverage_count[A_csc.indices[start:end]] += 1
                
            final_selected = []
            for b in selected_blocks:
                start = A_csc.indptr[b]
                end = A_csc.indptr[b+1]
                indices = A_csc.indices[start:end]
                
                # Temporarily remove
                coverage_count[indices] -= 1
                if not np.any(coverage_count < 1):
                    # Keep removed
                    pass
                else:
                    # Restore
                    coverage_count[indices] += 1
                    final_selected.append(b)
            
            best_solution = [list(all_blocks[i]) for i in final_selected]
            
    return best_solution
