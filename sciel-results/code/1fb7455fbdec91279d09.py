"""Strategy: Integer Linear Programming (ILP) formulation for Maximum Independent Set on the conflict graph.
We model the problem as selecting a subset of {2..256} such that no "forbidden triple" is fully selected.
This is a 3-uniform hypergraph independent set problem. We use scipy.optimize.milp to solve it exactly.
"""
import time
import sys
from math import isqrt
from itertools import combinations

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, isqrt(n) + 1):
        if n % i == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # Universe
    N = 256
    candidates = list(range(2, N + 1))
    n_vars = len(candidates)
    idx_map = {val: i for i, val in enumerate(candidates)}
    
    # 1. Identify all forbidden triples (a, b, c) distinct in candidates
    # where a divides b*c.
    # To speed up, we can precompute divisors or just check.
    # There are C(255, 3) ~ 2.7M triples. Checking each is fast enough in Python?
    # 2.7M * (1 mod op) ~ 0.1s. So yes.
    
    forbidden = []
    for i in range(n_vars):
        a = candidates[i]
        for j in range(i + 1, n_vars):
            b = candidates[j]
            for k in range(j + 1, n_vars):
                c = candidates[k]
                if (b * c) % a == 0:
                    forbidden.append((i, j, k))
    
    # 2. Formulate MILP
    # Maximize sum(x_i)
    # Subject to:
    #   x_i in {0, 1}
    #   x_i + x_j + x_k <= 2 for all (i, j, k) in forbidden
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Objective: maximize sum(x) -> minimize -sum(x)
        c = [-1.0] * n_vars
        
        # Constraints:
        # For each forbidden triple (i, j, k): x_i + x_j + x_k <= 2
        # This can be written as:
        #   x_i + x_j + x_k - 2 <= 0
        #   0 <= x_i + x_j + x_k <= 2 (since x_i are binary, sum is 0..3, we want to forbid 3)
        #   Actually, just x_i + x_j + x_k <= 2 is sufficient given 0 <= x_i <= 1.
        
        # We can build the A matrix for LinearConstraint.
        # A is sparse, but scipy.milp expects dense or sparse? 
        # scipy.sparse is available. Let's use it for memory efficiency if n_constraints is large.
        # Number of constraints = len(forbidden).
        # Max len(forbidden) is ~2.7M.
        # A matrix would be 2.7M x 255. 
        # Dense: 2.7M * 255 * 8 bytes ~ 5.5 GB. Too much.
        # Sparse: 2.7M * 3 non-zeros ~ 8.1M entries. ~ 65 MB. OK.
        
        import scipy.sparse as sp
        
        rows = []
        cols = []
        data = []
        
        for idx, (i, j, k) in enumerate(forbidden):
            rows.extend([idx, idx, idx])
            cols.extend([i, j, k])
            data.extend([1, 1, 1])
        
        A = sp.csr_matrix((data, (rows, cols)), shape=(len(forbidden), n_vars))
        
        # Constraint: A @ x <= 2
        # LinearConstraint(A, lb=0, ub=2)
        constraints = LinearConstraint(A, lb=0, ub=2)
        
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: x_i must be integer
        integrality = [1] * n_vars
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            x = res.x
            # x is a numpy array of floats (0.0 or 1.0)
            selected = [candidates[i] for i in range(n_vars) if x[i] > 0.5]
            # Convert to Python ints
            selected = [int(v) for v in selected]
            
            # Verify validity just in case (should be fast for ~50-60 elements)
            # If invalid, fall back to primes
            S = set(selected)
            valid = True
            for a in S:
                for b in S:
                    if a == b: continue
                    for c in S:
                        if a == c or b == c: continue
                        if (b * c) % a == 0:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                return selected
            else:
                # Fallback
                primes = [v for v in candidates if _is_prime(v)]
                return primes
        else:
            # Fallback if MILP fails
            primes = [v for v in candidates if _is_prime(v)]
            return primes
            
    except Exception as e:
        # Fallback to primes if anything goes wrong
        primes = [v for v in candidates if _is_prime(v)]
        return primes

# EVOLVE-BLOCK-END
