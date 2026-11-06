import math
import time
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin for safety
    
    # Domain: [2, 256]
    N = 256
    domain = list(range(2, N + 1))
    n = len(domain)
    
    # Weights: 1/x
    # Using float for scipy, but we can verify with Fraction at the end if needed.
    # However, scipy.milp expects float. The precision of double float is sufficient 
    # for distinguishing sums of reciprocals in this range (differences are > 1e-4 usually).
    weights = np.array([1.0 / x for x in domain], dtype=np.float64)
    
    # Build Conflict Matrix A
    # Constraint: For all pairs (i, j) with lcm(domain[i], domain[j]) <= 256, x_i + x_j <= 1
    # We need to identify all conflicting pairs.
    # n = 255. Pairs = ~32k. This is small.
    
    # Precompute lcm matrix or just check pairs
    # To speed up, we can precompute divisors or just use math.lcm.
    # math.lcm is C-implemented and fast.
    
    conflict_pairs = []
    for i in range(n):
        vi = domain[i]
        for j in range(i + 1, n):
            vj = domain[j]
            # Check lcm condition
            # lcm(vi, vj) = vi * vj // gcd(vi, vj)
            # If vi * vj // gcd(vi, vj) <= 256, they conflict.
            # Optimization: if vi > 256 or vj > 256, lcm > 256 (not possible here).
            # If vi * vj > 256 * max(vi, vj) then... no.
            # Just use math.lcm.
            if math.lcm(vi, vj) <= 256:
                conflict_pairs.append((i, j))
                
    # Check time
    if time.time() > deadline - 10:
        # Fallback: Greedy
        selected = []
        # Sort by weight desc (i.e., value asc)
        indices = sorted(range(n), key=lambda i: domain[i])
        for idx in indices:
            v = domain[idx]
            ok = True
            for other_idx in selected:
                if math.lcm(v, domain[other_idx]) <= 256:
                    ok = False
                    break
            if ok:
                selected.append(idx)
        return [domain[i] for i in selected]
        
    # Build A for scipy.optimize.milp
    # Number of constraints = len(conflict_pairs)
    m_constraints = len(conflict_pairs)
    
    if m_constraints == 0:
        return domain[:]
        
    # A is a sparse matrix of shape (m_constraints, n)
    # For each constraint (i, j), A[k, i] = 1, A[k, j] = 1
    # We can build a dense matrix if m_constraints * n is small enough.
    # m_constraints ~ 10k-20k? n=255. 20k * 255 = 5M floats. 40MB. Acceptable.
    # But sparse is better. scipy.milp accepts sparse.
    
    from scipy.sparse import lil_matrix
    A = lil_matrix((m_constraints, n), dtype=np.float64)
    
    for k, (i, j) in enumerate(conflict_pairs):
        A[k, i] = 1.0
        A[k, j] = 1.0
        
    # Convert to csc for efficiency
    A = A.tocsc()
    
    # Bounds for variables: x_i in [0, 1]
    # Since it's a set, x_i is binary.
    # scipy.optimize.milp allows integrality constraints.
    integrality = np.ones(n, dtype=np.int8) # 1 means integer
    
    # Objective: minimize -sum(weights * x)
    c = -weights
    
    # Constraints: A @ x <= 1
    # bounds: (0, 1) for all x
    # lb, ub can be provided as array or (lb, ub) tuple.
    # For milp, bounds are defined per variable.
    # bounds: list of (lb, ub) or array.
    # Let's use array for lb and ub if supported, or list.
    # scipy.optimize.milp signature: milp(c, constraints=..., integrality=..., bounds=...)
    # bounds can be a sequence of (lb, ub) pairs.
    bounds = [(0, 1)] * n
    
    try:
        from scipy.optimize import milp, LinearConstraint
        # Define constraint: A @ x <= 1
        # LinearConstraint(A, lb=-np.inf, ub=1)
        constraint = LinearConstraint(A, lb=-np.inf, ub=1)
        
        # Solve
        res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Success
            # res.x is the optimal solution
            # res.x might be 0.999999 or 1.000001 due to floating point.
            # Threshold at 0.5
            x_opt = res.x
            selected_indices = [i for i in range(n) if x_opt[i] > 0.5]
            
            if selected_indices:
                return [domain[i] for i in selected_indices]
            else:
                # Should not happen if weights are positive
                return [2]
        else:
            # Fallback if solver fails
            pass
            
    except Exception as e:
        pass
    
    # Fallback: Greedy Heuristic (Weighted)
    # Sort by 1/x descending (i.e., x ascending)
    # But we want to maximize sum.
    # Greedy: Pick element with highest weight (smallest x) that doesn't conflict with already picked.
    # This is a standard heuristic for MWIS.
    
    # Better Greedy: 
    # 1. Calculate "degree" or "conflict count" for each node.
    # 2. Or just simple greedy by weight.
    
    # Let's try a slightly smarter greedy: 
    # Sort by weight descending.
    # Iterate, add if no conflict.
    
    selected = []
    selected_set = set()
    
    # Sort indices by weight descending (value ascending)
    sorted_indices = sorted(range(n), key=lambda i: domain[i])
    
    for idx in sorted_indices:
        v = domain[idx]
        conflict = False
        for other_idx in selected:
            if math.lcm(v, domain[other_idx]) <= 256:
                conflict = True
                break
        if not conflict:
            selected.append(idx)
            selected_set.add(idx)
            
    if time.time() > deadline:
        return [domain[i] for i in selected]
        
    # Local Search / Improvement
    # Try to swap elements to increase sum?
    # Given the time, simple greedy is a safe fallback.
    # But let's try a few random restarts of greedy if time permits?
    # No, the ILP should have solved it. If it failed, something is wrong.
    
    # One more check: The ILP might be slow if the matrix is large.
    # But n=255 is tiny.
    
    return [domain[i] for i in selected]
