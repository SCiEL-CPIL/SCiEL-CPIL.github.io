"""
Solves the covering design problem C(v, k, t) using Integer Linear Programming (ILP).
Formulates the problem as minimizing the number of selected k-subsets such that 
every t-subset is covered.
"""
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

def construct(v, k, t):
    # Generate all possible t-subsets (constraints) and k-subsets (variables)
    t_subsets = list(itertools.combinations(range(v), t))
    k_subsets = list(itertools.combinations(range(v), k))
    
    n_constraints = len(t_subsets)
    n_variables = len(k_subsets)
    
    # Map t-subsets to indices for quick lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Build sparse incidence matrix A
    # A[i, j] = 1 if t_subsets[i] is a subset of k_subsets[j]
    rows = []
    cols = []
    data = []
    
    for j, k_block in enumerate(k_subsets):
        # Generate all t-subsets within this k_block
        for t_sub in itertools.combinations(k_block, t):
            i = t_subset_to_idx[t_sub]
            rows.append(i)
            cols.append(j)
            data.append(1)
            
    A = coo_matrix((data, (rows, cols)), shape=(n_constraints, n_variables))
    
    # Objective: minimize sum of x_j
    c = np.ones(n_variables)
    
    # Constraints: A x >= 1
    # milp requires constraints in the form lb <= A x <= ub
    # Here, lb = 1, ub = inf
    constraints = LinearConstraint(A, lb=1, ub=np.inf)
    
    # Bounds: x_j in [0, 1]
    bounds = Bounds(lb=0, ub=1)
    
    # Integrate variables
    integrality = np.ones(n_variables)
    
    # Solve ILP
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if not res.success:
        # Fallback to greedy if ILP fails or times out (though it should not for this size)
        return _greedy_fallback(v, k, t)
    
    # Extract selected blocks
    # res.x contains the solution vector. Values are close to 0 or 1.
    selected_indices = np.where(res.x > 0.5)[0]
    
    blocks = [list(k_subsets[i]) for i in selected_indices]
    
    # Ensure distinctness and sorting (though k_subsets are distinct and sorted by construction)
    # The problem requires blocks to be distinct. Since we select from a list of distinct combinations, they are distinct.
    # We sort each block just in case, though combinations are already sorted.
    for b in blocks:
        b.sort()
        
    return blocks

def _greedy_fallback(v, k, t):
    """
    Fallback greedy algorithm if ILP fails.
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
