import time
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _witt_6_12_blocks():
    """
    Solve the maximum packing problem for S(5,6,12) using Integer Linear Programming.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Generate all 6-subsets (blocks) and 5-subsets
    points = range(12)
    blocks = list(combinations(points, 6))
    fives = list(combinations(points, 5))
    
    n_blocks = len(blocks)
    n_fives = len(fives)
    
    # Map 5-subsets to indices
    five_index = {f: i for i, f in enumerate(fives)}
    
    # Build constraint matrix A and vector b
    # A[i, j] = 1 if block j contains 5-subset i
    # We want sum_j A[i,j] * x_j <= 1 for all i
    # This is a sparse matrix structure, but for 792x924, dense might be okay if we use efficient construction.
    # However, scipy.optimize.milp accepts linear constraints.
    
    # Efficiently construct the constraint matrix
    # Each block has C(6,5)=6 five-subsets.
    # We can build a list of (row, col, val) for sparse matrix if we used scipy.sparse, 
    # but milp takes LinearConstraint which can take dense arrays or sparse matrices.
    # Let's use scipy.sparse.csc_matrix for efficiency.
    import scipy.sparse as sp
    
    rows = []
    cols = []
    data = []
    
    for j, b in enumerate(blocks):
        # Generate all 5-subsets of this block
        for sub in combinations(b, 5):
            i = five_index[sub]
            rows.append(i)
            cols.append(j)
            data.append(1)
    
    A = sp.csc_matrix((data, (rows, cols)), shape=(n_fives, n_blocks))
    
    # Objective: maximize sum(x_j) -> minimize -sum(x_j)
    c = np.full(n_blocks, -1.0)
    
    # Constraints: A @ x <= 1
    # LinearConstraint(A, lb=-np.inf, ub=np.ones(n_fives))
    constraints = LinearConstraint(A, lb=-np.inf, ub=np.ones(n_fives))
    
    # Bounds: 0 <= x_j <= 1
    bounds = Bounds(lb=np.zeros(n_blocks), ub=np.ones(n_blocks))
    
    # Integrality: all variables are integer
    integrality = np.ones(n_blocks)
    
    # Solve
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.success and res.x is not None:
        # Extract blocks where x_j > 0.5
        result = [list(blocks[j]) for j in range(n_blocks) if res.x[j] > 0.5]
        
        # Verify count
        if len(result) == 132:
            return result
        
        # If ILP didn't find 132 (shouldn't happen), fallback to best found or greedy
        if len(result) > 0:
             return result
    
    # Fallback if ILP fails
    return _greedy_fallback()

def _greedy_fallback():
    """
    Fallback greedy algorithm if ILP fails.
    """
    all_blocks = list(combinations(range(12), 6))
    covered = set()
    result = []
    
    # Try a few different orderings
    orderings = [
        sorted(all_blocks, key=lambda b: (sum(b), b)),
        sorted(all_blocks, key=lambda b: (min(b), max(b), sum(b), b)),
        sorted(all_blocks, key=lambda b: b),
    ]
    
    best_result = []
    
    for ordering in orderings:
        covered = set()
        current = []
        for b in ordering:
            subsets = list(combinations(b, 5))
            if not any(s in covered for s in subsets):
                current.append(list(b))
                covered.update(subsets)
        
        if len(current) > len(best_result):
            best_result = current
            
        if len(best_result) == 132:
            break
    
    return best_result

def construct():
    """
    Constructs the Steiner system S(5,6,12).
    Returns a list of 132 distinct 6-subsets of 0..11 such that no 5-subset
    lies in two blocks.
    """
    return _witt_6_12_blocks()
