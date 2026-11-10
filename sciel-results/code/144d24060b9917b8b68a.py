import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    N = 256
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # 1. Generate all 3-term arithmetic progressions
    # A 3-AP is defined by (a, b, c) such that a + c = 2b and a < b < c
    # We can iterate by step d and start a.
    # a, a+d, a+2d
    # a >= 1, a+2d <= N
    # d >= 1
    # For a given d, a can go from 1 to N - 2d.
    
    ap_constraints = []
    for d in range(1, (N + 1) // 2):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            # The constraint is x_a + x_b + x_c <= 2
            # We represent this as a row in the constraint matrix
            # But building a dense 16000x256 matrix is fine (4M floats ~ 32MB)
            # However, sparse is better. scipy.optimize.milp accepts sparse constraints?
            # LinearConstraint accepts dense arrays or sparse matrices.
            # Let's collect indices for sparse construction.
            ap_constraints.append((a-1, b-1, c-1)) # 0-indexed

    num_constraints = len(ap_constraints)
    
    # Build sparse matrix A for constraints
    # A @ x <= 2
    # Each constraint has 3 non-zeros (value 1)
    row_indices = []
    col_indices = []
    data = []
    
    for i, (a, b, c) in enumerate(ap_constraints):
        row_indices.extend([i, i, i])
        col_indices.extend([a, b, c])
        data.extend([1.0, 1.0, 1.0])
        
    from scipy.sparse import csr_matrix
    A = csr_matrix((data, (row_indices, col_indices)), shape=(num_constraints, N))
    
    # Objective: Maximize sum(1/i * x_i) -> Minimize -sum(1/i * x_i)
    c = [-recip[i] for i in range(1, N + 1)]
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integer
    integrality = np.ones(N)
    
    # Constraint: A @ x <= 2
    # LinearConstraint(A, lb=-np.inf, ub=2)
    constraint = LinearConstraint(A, lb=-np.inf, ub=np.ones(num_constraints) * 2)
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=[constraint],
            integrality=integrality,
            bounds=bounds,
            options={'time_limit': 80.0}
        )
        
        # Check if solution is found
        if res.status == 0 or res.status == 1: # 0 is optimal, 1 is feasible
            x = res.x
            # x is a numpy array of 0s and 1s (floats)
            # Rounding might be needed if not exact, but integrality should handle it
            # Threshold 0.5
            selected = [i + 1 for i in range(N) if x[i] > 0.5]
            
            # Validate
            S_set = set(selected)
            valid = True
            for a in S_set:
                for b in S_set:
                    if a >= b:
                        continue
                    s = a + b
                    if s % 2 == 0:
                        mid = s // 2
                        if mid in S_set and mid != a and mid != b:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid and len(selected) > 0:
                return selected
            else:
                # If invalid, fall through to fallback
                pass
                
    except Exception as e:
        # If solver fails, fall through
        pass

    # Fallback: Greedy heuristic from seed code
    # This ensures we always return a valid non-empty set
    
    conflict_pairs = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            c = 2 * b - a
            if c > N or c <= b:
                continue
            conflict_pairs[a].append((b, c))
            conflict_pairs[b].append((a, c))
            conflict_pairs[c].append((a, b))

    def can_add(x, S_mask):
        for y, z in conflict_pairs[x]:
            if (S_mask & (1 << y)) and (S_mask & (1 << z)):
                return False
        return True

    def greedy_fill(forbidden_mask, S_mask, order=None):
        S = S_mask
        if order is None:
            order = range(1, N + 1)
        for x in order:
            if forbidden_mask & (1 << x):
                continue
            if S & (1 << x):
                continue
            if can_add(x, S):
                S |= (1 << x)
        return S

    # Try descending order (prioritize small numbers with high reciprocals)
    S_desc = greedy_fill(0, 0, order=range(N, 0, -1))
    result = [x for x in range(1, N + 1) if S_desc & (1 << x)]
    
    # Validate fallback
    S_set = set(result)
    valid = True
    for a in S_set:
        for b in S_set:
            if a >= b:
                continue
            s = a + b
            if s % 2 == 0:
                mid = s // 2
                if mid in S_set and mid != a and mid != b:
                    valid = False
                    break
        if not valid:
            break
            
    if not valid:
        # Absolute fallback: powers of 2
        result = [2**i for i in range(8) if 2**i <= N]
        
    return result
