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
    rows = []
    cols = []
    
    for j, block in enumerate(all_blocks):
        for t_sub in itertools.combinations(block, t):
            i = t_subset_to_idx[t_sub]
            rows.append(i)
            cols.append(j)
            
    A = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_t_subsets, n_blocks))
    
    # 1. Solve LP Relaxation to get weights
    # Minimize sum(x) subject to A x >= 1, 0 <= x <= 1
    c = np.ones(n_blocks)
    constraints = LinearConstraint(A, 1, np.inf)
    bounds = Bounds(0, 1)
    integrality = np.zeros(n_blocks) # LP relaxation
    
    x_lp = None
    try:
        start_time = time.time()
        res_lp = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, 
                      options={"time_limit": 5})
        if res_lp.x is not None:
            x_lp = res_lp.x
        # If LP fails or takes too long, x_lp remains None or we use uniform weights
    except Exception:
        x_lp = np.ones(n_blocks) / n_blocks # Fallback: uniform weights
        
    if x_lp is None:
        x_lp = np.ones(n_blocks) / n_blocks

    # 2. Greedy Heuristic with LP-guided tie-breaking
    uncovered = np.ones(n_t_subsets, dtype=bool)
    selected_blocks = []
    
    # Precompute block_covers for fast access
    block_covers = [[] for _ in range(n_blocks)]
    for i_idx, i in enumerate(rows):
        block_covers[cols[i_idx]].append(i)
        
    A_T = A.T.tocsr()
    
    # To speed up tie-breaking, we can't easily do it in argmax.
    # We will find all blocks with max gain, then pick the one with highest x_lp.
    # Or, we can modify the gain calculation? No, gain is integer count.
    
    while np.any(uncovered):
        # Calculate gain for all blocks
        gains = A_T @ uncovered.astype(np.float32)
        
        # Find max gain
        max_gain = np.max(gains)
        
        if max_gain == 0:
            break # Should not happen
            
        # Find all blocks with max gain
        candidates = np.where(gains == max_gain)[0]
        
        # Pick the candidate with the highest LP weight
        if len(candidates) > 1:
            best_block = candidates[np.argmax(x_lp[candidates])]
        else:
            best_block = candidates[0]
            
        selected_blocks.append(best_block)
        
        # Update uncovered
        for ts_idx in block_covers[best_block]:
            uncovered[ts_idx] = False
            
    # 3. Post-processing: Remove redundant blocks (Deletion)
    coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
    for b in selected_blocks:
        for ts_idx in block_covers[b]:
            coverage_count[ts_idx] += 1
            
    final_selected = []
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

    # 4. Local Search: 1-Exchange (Swap) to try to reduce size?
    # Actually, deletion already removes redundant ones.
    # To reduce the count further, we need to find a block that can replace two others?
    # That's complex.
    # Let's try a simple improvement: 
    # Try to swap a block in the solution with a block not in the solution 
    # such that the new solution is valid and has the same size (no gain in count) 
    # but maybe allows further deletions?
    # Given time, we'll stick to the LP-guided greedy + deletion.
    
    blocks = [list(all_blocks[i]) for i in final_selected]
    
    return blocks
