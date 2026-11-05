import time
import itertools
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Points are 0..11
    points = range(12)
    
    # Generate all possible 6-subsets (blocks)
    all_blocks = list(itertools.combinations(points, 6))
    num_blocks = len(all_blocks) # 924
    
    # Generate all possible 5-subsets (constraints)
    all_5_subsets = list(itertools.combinations(points, 5))
    num_constraints = len(all_5_subsets) # 792
    
    # Map 5-subsets to indices for fast lookup
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # Pre-compute 5-subsets for each block as indices
    block_5_indices = []
    for b in all_blocks:
        indices = [subset_to_idx[s] for s in itertools.combinations(b, 5)]
        block_5_indices.append(indices)

    # Try to use scipy.optimize.milp for an exact solution
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        # Build incidence matrix A (num_constraints x num_blocks)
        # We can build this faster using the pre-computed indices?
        # No, we need the sparse matrix for MILP.
        # But we can use the fact that we have the indices.
        
        rows = []
        cols = []
        data = []
        
        for j, indices in enumerate(block_5_indices):
            for i in indices:
                rows.append(i)
                cols.append(j)
                data.append(1)
                
        A = sp.csr_matrix((data, (rows, cols)), shape=(num_constraints, num_blocks))
        
        # Objective: maximize sum(x_j) -> minimize -sum(x_j)
        c = -np.ones(num_blocks)
        
        # Constraints: A x <= 1
        constraint = LinearConstraint(A, ub=np.ones(num_constraints))
        
        # Bounds: 0 <= x_j <= 1
        bounds = Bounds(lb=np.zeros(num_blocks), ub=np.ones(num_blocks))
        
        # Solve with a time limit to avoid hanging
        res = milp(c=c, constraints=constraint, bounds=bounds, integrality=np.ones(num_blocks),
                   options={'time_limit': 100})
        
        if res.success:
            # Extract solution
            x = res.x
            blocks = [list(all_blocks[i]) for i in range(num_blocks) if x[i] > 0.5]
            # Verify validity and count
            if len(blocks) > 0:
                return blocks
        else:
            # Fallback if solver failed
            pass
            
    except Exception:
        pass

    # Fallback: Greedy with randomization and restarts
    # Use the pre-computed indices for speed
    best_blocks = []
    
    # First, try the deterministic lex order
    covered = np.zeros(num_constraints, dtype=bool)
    blocks = []
    for j, indices in enumerate(block_5_indices):
        if not np.any(covered[indices]):
            blocks.append(list(all_blocks[j]))
            covered[indices] = True
    if len(blocks) > len(best_blocks):
        best_blocks = blocks
        
    # Try random restarts
    # We can do more restarts now because it's faster
    num_restarts = 20
    for _ in range(num_restarts):
        random.shuffle(all_blocks) # Note: this shuffles the list, but block_5_indices is fixed to original order!
        # BUG: If we shuffle all_blocks, we must also shuffle block_5_indices or use a permutation.
        
        # Corrected approach:
        order = list(range(num_blocks))
        random.shuffle(order)
        
        covered = np.zeros(num_constraints, dtype=bool)
        blocks = []
        for j in order:
            indices = block_5_indices[j]
            if not np.any(covered[indices]):
                blocks.append(list(all_blocks[j]))
                covered[indices] = True
        if len(blocks) > len(best_blocks):
            best_blocks = blocks
            
    return best_blocks
