import itertools
import time
import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    points = list(range(v))
    
    # 1. Generate all t-subsets and k-blocks
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # 2. Build sparse incidence matrix A (num_tsubs x num_kblocks)
    # A[i, j] = 1 if tsub i is contained in kblock j
    row_indices = []
    col_indices = []
    data = []
    
    for j, block in enumerate(all_kblocks):
        # Find all t-subsets of this block
        for ts in itertools.combinations(block, t):
            i = tsub_to_idx[ts]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = csc_matrix((data, (row_indices, col_indices)), shape=(num_tsubs, num_kblocks))
    
    # 3. Greedy Construction
    covered = np.zeros(num_tsubs, dtype=bool)
    x0 = np.zeros(num_kblocks, dtype=float)
    selected_blocks = []
    
    # Precompute t-subset indices for each block for fast updating
    # Actually, we can just use the matrix structure or precomputed lists
    # Precomputed lists are faster for updating covered array
    block_tsub_idxs = []
    for j, block in enumerate(all_kblocks):
        idxs = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsub_idxs.append(idxs)
        
    while time.monotonic() < deadline - 5.0: # Leave time for ILP
        u = ~covered
        gains = A.T @ u
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        
        selected_blocks.append(best_idx)
        x0[best_idx] = 1.0
        
        # Update covered
        for idx in block_tsub_idxs[best_idx]:
            covered[idx] = True
            
        if np.all(covered):
            break
            
    # 4. Polish (Remove redundant blocks)
    # We have a list of selected block indices
    # We want to remove blocks if possible
    # Maintain count of coverage for each t-subset
    counts = np.zeros(num_tsubs, dtype=int)
    for b_idx in selected_blocks:
        for idx in block_tsub_idxs[b_idx]:
            counts[idx] += 1
            
    # Try to remove blocks
    # Iterate multiple times
    changed = True
    while changed and time.monotonic() < deadline - 10.0:
        changed = False
        for b_idx in selected_blocks[:]:
            # Check if removing b_idx makes any count 0
            # For all tsubs in b_idx, if counts[tsub] == 1, we can't remove
            can_remove = True
            for idx in block_tsub_idxs[b_idx]:
                if counts[idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove b_idx
                selected_blocks.remove(b_idx)
                x0[b_idx] = 0.0
                for idx in block_tsub_idxs[b_idx]:
                    counts[idx] -= 1
                changed = True
                
    # 5. ILP Optimization
    # Only if we have time
    if time.monotonic() < deadline - 15.0:
        try:
            # Prepare ILP
            c = np.ones(num_kblocks)
            # Constraints: A x >= 1
            # A is sparse
            constraints = LinearConstraint(A, 1, np.inf)
            integrality = np.ones(num_kblocks)
            bounds = Bounds(0, 1)
            
            # x0 is the current solution
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                x0=x0,
                time_limit=10.0 # Use remaining time, maybe less
            )
            
            # Check result
            if res.x is not None:
                # Round to nearest integer
                x_sol = np.round(res.x)
                # Validate
                # Check if x_sol is a valid cover
                # Sum A @ x_sol >= 1
                # And x_sol is binary
                if np.all((x_sol == 0) | (x_sol == 1)):
                    coverage = A @ x_sol
                    if np.all(coverage >= 1):
                        # Count blocks
                        new_count = np.sum(x_sol)
                        old_count = len(selected_blocks)
                        if new_count < old_count:
                            # Update selected_blocks
                            selected_blocks = [i for i in range(num_kblocks) if x_sol[i] == 1]
                            
        except Exception:
            pass # Fall back to greedy+polish
            
    # Convert indices back to lists
    result_blocks = [list(all_kblocks[i]) for i in selected_blocks]
    
    return result_blocks
