"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin from 120s limit
    
    # Generate all possible 6-subsets
    points = range(12)
    all_blocks = list(combinations(points, 6))
    n_blocks = len(all_blocks) # 924
    
    # Generate all 5-subsets
    all_5subsets = list(combinations(points, 5))
    n_5subsets = len(all_5subsets) # 792
    
    # Map 5-subsets to indices
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    # Build constraint matrix A: A[i, j] = 1 if 5-subset i is in block j
    # We want to maximize sum(x_j) subject to A @ x <= 1
    # A is 792 x 924
    
    # Using scipy.optimize.milp
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        # Create sparse matrix
        rows = []
        cols = []
        data = []
        for j, b in enumerate(all_blocks):
            # Get 5-subsets of this block
            for s in combinations(b, 5):
                i = subset_to_idx[s]
                rows.append(i)
                cols.append(j)
                data.append(1)
        
        A = sp.csr_matrix((data, (rows, cols)), shape=(n_5subsets, n_blocks))
        
        # Objective: maximize sum(x) -> minimize -sum(x)
        c = -np.ones(n_blocks)
        
        # Constraints: A @ x <= 1
        # milp expects: lb <= A @ x <= ub
        constraint = LinearConstraint(A, lb=None, ub=np.ones(n_5subsets))
        
        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # integrality: 1 for all variables
        integrality = np.ones(n_blocks)
        
        # Solve
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status == 0 and res.success:
            x = res.x
            # Filter selected blocks
            selected = [all_blocks[j] for j in range(n_blocks) if x[j] > 0.5]
            # Verify count
            if len(selected) > 0:
                return [list(b) for b in selected]
    except Exception as e:
        pass # Fall through to heuristic if MILP fails
    
    # Fallback: Greedy with random restarts
    # Sort blocks by some heuristic or just random
    import random
    best_blocks = []
    best_count = 0
    
    # Try a few random orderings
    for _ in range(50):
        if time.time() > deadline:
            break
        random.shuffle(all_blocks)
        covered = set()
        current_blocks = []
        for b in all_blocks:
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                current_blocks.append(b)
                covered.update(subsets)
        if len(current_blocks) > best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            if best_count == 132:
                break
        
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
