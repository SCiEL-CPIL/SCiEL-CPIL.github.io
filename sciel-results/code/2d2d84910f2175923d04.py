import itertools
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    row_indices = []
    col_indices = []
    data = []
    
    # Precompute covered t-subsets for each block for faster access later
    block_t_subsets = []
    for i, block in enumerate(all_blocks):
        subs = list(itertools.combinations(block, t))
        block_t_subsets.append(subs)
        for t_sub in subs:
            j = t_subset_to_idx[t_sub]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = csr_matrix((data, (row_indices, col_indices)), shape=(num_blocks, num_t_subsets))
    
    # Phase 1: Greedy Set Cover
    active = np.ones(num_t_subsets, dtype=np.float64)
    selected_block_indices = []
    
    while True:
        if time.time() > deadline - 60.0:
            break
            
        gains = A.dot(active)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        b_idx = np.argmax(gains)
        selected_block_indices.append(b_idx)
        
        covered_indices = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        active[covered_indices] = 0
        
        if np.all(active == 0):
            break
            
    # Phase 2: Local Search Deletion
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
        
    changed = True
    while changed:
        changed = False
        to_remove = []
        for i, b_idx in enumerate(selected_block_indices):
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            redundant = True
            for c in covered:
                if cover_counts[c] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
        
        if to_remove:
            for i in to_remove:
                b_idx = selected_block_indices[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
            remove_set = set(to_remove)
            selected_block_indices = [b for i, b in enumerate(selected_block_indices) if i not in remove_set]
            changed = True
            
    current_best = len(selected_block_indices)
    
    # Phase 3: Try ILP refinement on the current solution
    # We try to replace some blocks with fewer blocks.
    # Strategy: Keep a subset of blocks fixed, try to cover the rest with fewer blocks.
    # Or, simpler: Try to solve the ILP for the whole problem but with a time limit,
    # starting from the greedy solution as a warm start (if supported) or just using the
    # greedy solution to bound the objective.
    
    # Since scipy.milp doesn't support warm starts easily, we will try to solve the ILP
    # directly but with a tight time limit. If it finds a better solution, we use it.
    # However, solving the full ILP might be too slow.
    
    # Alternative: Use the greedy solution to identify "critical" t-subsets.
    # Let's try a simpler ILP approach:
    # Minimize sum(x) subject to A^T x >= 1, x in {0,1}
    
    # To make it tractable, we can limit the search space to blocks that are "useful".
    # A block is useful if it covers at least one t-subset that is not covered by the
    # majority of other blocks in the greedy solution, or simply if it was selected.
    
    # Let's try to solve the ILP on the full space but with a very short time limit.
    # If it doesn't finish, we ignore the result.
    
    if time.time() < deadline - 20.0:
        try:
            # Objective: minimize sum(x)
            c = np.ones(num_blocks)
            
            # Constraints: A^T x >= 1
            # A is (num_blocks, num_t_subsets), so A^T is (num_t_subsets, num_blocks)
            A_T = A.T.tocsr()
            
            constraint = LinearConstraint(A_T, lb=1, ub=np.inf)
            
            # Bounds: x in [0, 1]
            lb = np.zeros(num_blocks)
            ub = np.ones(num_blocks)
            bounds = Bounds(lb, ub)
            
            # Integrality: all variables are binary
            integrality = np.ones(num_blocks)
            
            # Run MILP with a time limit
            # Note: scipy.optimize.milp uses HiGHS by default
            # We need to pass options to limit time
            # However, scipy.milp interface doesn't directly expose time limit in a standard way
            # across all versions. We rely on the deadline check outside or just let it run
            # and hope it's fast enough or fails fast.
            
            # Actually, scipy.optimize.milp has a `options` parameter.
            # We can try to set time_limit.
            
            res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds,
                       options={'time_limit': 10.0})
            
            if res.status == 0 and res.x is not None:
                ilp_solution = res.x.astype(int)
                ilp_blocks = np.where(ilp_solution == 1)[0]
                
                # Verify validity
                if len(ilp_blocks) > 0:
                    cover_check = np.zeros(num_t_subsets, dtype=np.int32)
                    for b_idx in ilp_blocks:
                        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                        cover_check[covered] += 1
                    
                    if np.all(cover_check >= 1) and len(ilp_blocks) < current_best:
                        selected_block_indices = list(ilp_blocks)
                        current_best = len(selected_block_indices)
                        
        except Exception:
            pass
            
    # Phase 4: Final Deletion Pass on the best solution found
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
        
    changed = True
    while changed:
        changed = False
        to_remove = []
        for i, b_idx in enumerate(selected_block_indices):
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            redundant = True
            for c in covered:
                if cover_counts[c] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
        
        if to_remove:
            for i in to_remove:
                b_idx = selected_block_indices[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
            remove_set = set(to_remove)
            selected_block_indices = [b for i, b in enumerate(selected_block_indices) if i not in remove_set]
            changed = True
            
    # Convert to output format
    blocks = [list(all_blocks[b_idx]) for b_idx in selected_block_indices]
    return blocks
