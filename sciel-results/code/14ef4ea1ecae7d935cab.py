import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
import itertools
import time

def construct(v, k, t):
    # Parameters
    # Precompute all k-subsets (blocks) and t-subsets (coverage requirements)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map t-subsets to indices for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build sparse incidence matrix A
    # A[i, j] = 1 if t_subset i is contained in block j
    # We build it in COO format for efficiency
    rows = []
    cols = []
    
    for j, block in enumerate(all_blocks):
        for t_sub in itertools.combinations(block, t):
            i = t_subset_to_idx[t_sub]
            rows.append(i)
            cols.append(j)
            
    A = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_t_subsets, n_blocks))
    
    # ILP Formulation
    # Minimize sum(x_j)
    # Subject to: A @ x >= 1
    # x_j in {0, 1}
    
    c = np.ones(n_blocks)
    constraints = LinearConstraint(A, 1, np.inf)
    bounds = Bounds(0, 1)
    integrality = np.ones(n_blocks)
    
    start_time = time.time()
    ilp_solution = None
    
    try:
        # Run ILP with a time limit that leaves buffer for fallback
        # 110 seconds is safe given 120s total limit
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 110, "mip_rel_gap": 0.0})
        
        # Check if a feasible solution was found
        # Status 0: optimal, 1: feasible (not necessarily optimal), 2: infeasible, etc.
        if res.status in [0, 1] and res.x is not None:
            x_int = np.round(res.x).astype(int)
            # Verify validity just in case
            if np.all(A @ x_int >= 1):
                ilp_solution = [list(all_blocks[i]) for i in np.where(x_int == 1)[0]]
                
    except Exception:
        pass
        
    if ilp_solution is not None:
        return ilp_solution
        
    # Fallback: Fast Greedy Heuristic
    # This is a new implementation using numpy for speed, distinct from the previous loop-based one.
    # We maintain a count of uncovered t-subsets.
    
    # Precompute which t-subsets each block covers
    # Using the rows/cols from A construction
    # A is CSR. We can extract column data if we convert to CSC, or just use the lists.
    # Let's create block_covers as a list of numpy arrays for fast vectorized updates?
    # Or just use the lists from A's construction.
    
    # Re-use rows/cols lists from above if still in scope? Yes.
    # But they are local to the previous block? No, they are in function scope.
    
    # Create block_covers: list of lists of t_subset indices
    block_covers = [[] for _ in range(n_blocks)]
    for r, c_idx in zip(rows, cols):
        block_covers[c_idx].append(r)
        
    # Convert to numpy arrays for faster iteration if needed, or keep lists.
    # Lists are fine for the greedy loop if we are careful.
    
    uncovered_count = np.zeros(n_t_subsets, dtype=np.int32)
    # Initially, all are "uncovered" in the sense that they need to be covered.
    # Actually, for greedy, we want to know how many *new* t-subsets a block covers.
    # Let's maintain `covered` boolean array.
    covered = np.zeros(n_t_subsets, dtype=bool)
    
    selected_blocks = []
    
    # Precompute gain for each block initially?
    # Gain = number of t-subsets in block that are currently uncovered.
    # Initially, all are uncovered. So gain[j] = len(block_covers[j]) = C(k, t).
    # But we need to update gains dynamically.
    
    # Efficient Greedy:
    # 1. Calculate initial gains.
    gains = np.array([len(bc) for bc in block_covers], dtype=np.int32)
    
    while True:
        if np.all(covered):
            break
            
        # Find block with max gain
        best_block = int(np.argmax(gains))
        
        if gains[best_block] == 0:
            # No block can cover any new t-subset? This should not happen if a solution exists.
            # If it does, we break to avoid infinite loop.
            break
            
        selected_blocks.append(best_block)
        
        # Update covered status and gains for other blocks
        # For each t_subset in best_block:
        #   if not covered:
        #     covered = True
        #     for each block j covering this t_subset:
        #       gains[j] -= 1
        
        # To do this efficiently, we need to know which blocks cover each t_subset.
        # This is the transpose of the incidence matrix.
        # Let's precompute `t_subset_to_blocks`: list of lists of block indices.
        # This might be memory heavy but v=19 is small.
        
        # Let's construct `t_subset_to_blocks` once.
        if not hasattr(construct, '_t_to_blocks'):
             construct._t_to_blocks = [[] for _ in range(n_t_subsets)]
             for r, c_idx in zip(rows, cols):
                 construct._t_to_blocks[r].append(c_idx)
                 
        t_to_blocks = construct._t_to_blocks
        
        for ts_idx in block_covers[best_block]:
            if not covered[ts_idx]:
                covered[ts_idx] = True
                # Update gains for all blocks covering this t_subset
                for j in t_to_blocks[ts_idx]:
                    gains[j] -= 1
                    
    # Post-processing: Remove redundant blocks
    # Check if removing a block leaves all its t-subsets covered by others.
    final_selected = []
    
    # Count coverage for current selection
    coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
    for b in selected_blocks:
        for ts_idx in block_covers[b]:
            coverage_count[ts_idx] += 1
            
    for b in selected_blocks:
        # Temporarily remove b
        for ts_idx in block_covers[b]:
            coverage_count[ts_idx] -= 1
            
        if np.any(coverage_count < 1):
            # Restore b
            for ts_idx in block_covers[b]:
                coverage_count[ts_idx] += 1
            final_selected.append(b)
            
    blocks = [list(all_blocks[i]) for i in final_selected]
    
    return blocks
