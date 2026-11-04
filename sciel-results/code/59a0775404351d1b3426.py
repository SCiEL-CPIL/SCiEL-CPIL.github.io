import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
import itertools
import time

def construct(v, k, t):
    # Parameters
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map t-subsets to indices
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence matrix A (n_t_subsets x n_blocks)
    # A[i, j] = 1 if t_subset i is in block j
    # Using sparse matrix
    rows = []
    cols = []
    
    for j, block in enumerate(all_blocks):
        for t_sub in itertools.combinations(block, t):
            i = t_subset_to_idx[t_sub]
            rows.append(i)
            cols.append(j)
            
    A = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_t_subsets, n_blocks))
    
    # Fast Greedy Heuristic
    # uncovered: boolean array of length n_t_subsets
    uncovered = np.ones(n_t_subsets, dtype=bool)
    selected_blocks = []
    
    # Precompute the non-zero indices for each column of A (i.e., t-subsets covered by each block)
    # A is CSR, so A.indices and A.indptr give row indices for each column?
    # Wait, CSR is Row-based. We need Column-based for "which t-subsets does block j cover?".
    # Let's convert to CSC or just use the rows/cols lists.
    # Actually, let's stick to the rows/cols lists for speed in construction, but for fast lookup in greedy,
    # we need to know which t-subsets a block covers.
    
    # Let's create a list of lists: block_covers[j] = list of t_subset indices
    block_covers = [[] for _ in range(n_blocks)]
    for i_idx, i in enumerate(rows):
        block_covers[cols[i_idx]].append(i)
        
    # Greedy Loop
    # To speed up "max gain", we can maintain an array `gain` of length n_blocks.
    # gain[j] = number of currently uncovered t-subsets covered by block j.
    # Initially, gain[j] = len(block_covers[j]) = C(k,t) for all j.
    # When we select a block b, we set uncovered[ts] = False for ts in block_covers[b].
    # Then for all other blocks j, if ts in block_covers[j], gain[j] -= 1.
    # This update is expensive: sum of len(block_covers[j]) for all j that share a t-subset with b.
    # Average overlap?
    # This "dynamic gain" approach is $O(N_{blocks} \times \text{avg\_overlap})$.
    # Simpler: Just recompute gain for all blocks each step using A.T @ uncovered.
    # A is sparse. A.T is CSC (or CSR if we transpose).
    # A.T @ uncovered is a vector of length n_blocks.
    # This is a sparse matrix vector multiply. $O(\text{nnz}(A)) \approx 2.7 \times 10^6$ ops.
    # 200 steps -> $5.4 \times 10^8$ ops. NumPy/SciPy sparse matvec is fast.
    
    A_T = A.T.tocsr() # Transpose and convert to CSR for efficient row access? No, matvec works on any sparse format.
    
    while np.any(uncovered):
        # Calculate gain for all blocks
        gains = A_T @ uncovered.astype(np.float32)
        
        # Find block with max gain
        best_block = int(np.argmax(gains))
        
        if gains[best_block] == 0:
            break # Should not happen if uncovered is not empty and solution exists
            
        selected_blocks.append(best_block)
        
        # Update uncovered
        # Get indices of t-subsets covered by best_block
        # In CSR A, column best_block corresponds to rows in A.indices[A.indptr[best_block]:A.indptr[best_block+1]]?
        # No, A is CSR. A[j] is row j. We need column best_block.
        # Let's use the block_covers list created earlier.
        for ts_idx in block_covers[best_block]:
            uncovered[ts_idx] = False
            
    # Post-processing: Remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other selected blocks.
    # We can check this by:
    # 1. Build a matrix of selected blocks.
    # 2. For each selected block, check if removing it leaves all its t-subsets covered.
    
    final_selected = []
    # To make this fast, we can iterate and check.
    # Let's create a coverage count for the current selection.
    # coverage_count[ts] = number of selected blocks covering ts.
    
    coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
    for b in selected_blocks:
        for ts_idx in block_covers[b]:
            coverage_count[ts_idx] += 1
            
    # Now, try to remove each block
    # This is O(N_selected * C(k,t)).
    # N_selected ~ 200. C(k,t)=35. 7000 ops. Very fast.
    
    for b in selected_blocks:
        # Temporarily remove b
        for ts_idx in block_covers[b]:
            coverage_count[ts_idx] -= 1
            
        # Check if any t-subset is now uncovered
        if np.any(coverage_count < 1):
            # Restore b
            for ts_idx in block_covers[b]:
                coverage_count[ts_idx] += 1
            final_selected.append(b)
        else:
            # Keep b removed
            pass

    # Convert indices to blocks
    blocks = [list(all_blocks[i]) for i in final_selected]
    
    # Now, try ILP to improve?
    # Given time constraints and previous failure, I will attempt a short ILP run.
    # If it finds a solution better than the greedy one, return it.
    
    current_score = len(blocks)
    
    # ILP Setup
    c = np.ones(n_blocks)
    constraints = LinearConstraint(A, 1, np.inf)
    bounds = Bounds(0, 1)
    integrality = np.ones(n_blocks)
    
    start_time = time.time()
    try:
        # Limit time to 10s to ensure we have time for fallback if needed, 
        # though greedy already returned a valid solution.
        # If ILP finds optimal, it's great. If not, we stick with greedy.
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 10, "mip_rel_gap": 0.0})
        
        if res.status == 0 or res.status == 1:
            if res.x is not None:
                x_int = np.round(res.x).astype(int)
                if np.all(A @ x_int >= 1):
                    ilp_blocks = [list(all_blocks[i]) for i in np.where(x_int == 1)[0]]
                    if len(ilp_blocks) < current_score:
                        return ilp_blocks
                        
    except Exception:
        pass
        
    return blocks
