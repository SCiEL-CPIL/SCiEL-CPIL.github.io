import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction

def construct():
    start_time = time.time()
    # Safety margin: aim to finish well before 90s. 
    # MIP solvers can be unpredictable, so we use a conservative deadline.
    # If we run out of time, we return the best incumbent found so far.
    deadline = start_time + 80.0
    
    n = 256
    
    # 1. Precompute all 3-term arithmetic progressions (a, b, c) with a < b < c and a, b, c in [1, n]
    # Constraint: x_a + x_b + x_c <= 2
    # We store these as indices (0-based for scipy)
    triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            c = 2 * b - a
            if c > b and c <= n:
                triples.append((a - 1, b - 1, c - 1))
                
    num_triples = len(triples)
    
    # 2. Define the objective function
    # Maximize sum(1/i) * x_i
    # scipy.optimize.milp minimizes, so we use -1/i
    # To avoid floating point issues in the solver, we can scale weights or use high precision.
    # However, standard float64 is usually sufficient for n=256.
    weights = np.array([-1.0 / i for i in range(1, n + 1)])
    
    # 3. Define constraints
    # For each triple (i, j, k), we have x_i + x_j + x_k <= 2
    # This can be written as a matrix A_ub such that A_ub @ x <= b_ub
    # A_ub will be num_triples x n
    
    # For efficiency, we use a sparse matrix if available, but scipy.milp accepts dense or sparse.
    # Given n=256 and ~10k triples, dense is 10k * 256 * 8 bytes ~ 20MB, which is fine.
    A_ub = np.zeros((num_triples, n))
    b_ub = np.full(num_triples, 2.0)
    
    for idx, (i, j, k) in enumerate(triples):
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        A_ub[idx, k] = 1.0
        
    # Variable bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0.0, ub=1.0)
    
    # Integrality: all variables are binary (0 or 1)
    integrality = np.ones(n)
    
    # Linear constraint object
    # A_ub @ x <= b_ub
    # Note: scipy.milp uses LinearConstraint(A, lb, ub)
    # Here A is A_ub, lb is -inf (or very small), ub is b_ub
    # Actually, LinearConstraint expects A @ x >= lb and A @ x <= ub.
    # So we pass lb=None (or -np.inf) and ub=b_ub.
    constraints = LinearConstraint(A_ub, lb=None, ub=b_ub)
    
    # 4. Solve
    # We want to maximize, so we minimize -weights.
    # result.fun will be the minimum value of the objective.
    # The actual score is -result.fun.
    
    # Check time before starting? No, just start and monitor.
    # However, milp is a blocking call. We can't easily interrupt it mid-solve in Python without threads.
    # Given n=256, the problem size is moderate. 
    # Number of variables: 256.
    # Number of constraints: ~ (256*255/2) / 2 approx? 
    # Let's estimate triples count.
    # For each difference d, number of APs is n - 2d.
    # Sum_{d=1}^{n/2} (n - 2d) ~ n^2/4 - n^2/4 = small? No.
    # Sum (n-2d) for d=1 to 128.
    # 128*n - 2*(128*129/2) = 32768 - 16512 = 16256.
    # So ~16k constraints.
    # 256 vars, 16k constraints. This is a small MIP for modern solvers (HiGHS).
    # It should solve in seconds.
    
    try:
        # time_limit is in seconds. We set it slightly below our deadline to allow for overhead.
        # But wait, if we set time_limit, HiGHS will stop and return incumbent.
        # We want to use as much time as possible to prove optimality or find better incumbent.
        # Let's set time_limit to (deadline - current_time) - 2.0
        current_time = time.time()
        remaining = deadline - current_time - 2.0
        if remaining < 1.0:
             # Fallback if we are already too late
             return [1]
             
        res = milp(
            c=weights,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": remaining, "mip_rel_gap": 1e-9} # Tight gap for precision
        )
        
        if res.success and res.x is not None:
            # res.x contains the optimal (or best found) solution
            # Convert to list of integers
            solution_indices = np.where(res.x > 0.5)[0]
            # Convert back to 1-based indices
            S = sorted([int(i) + 1 for i in solution_indices])
            
            # Sanity check: Ensure no 3-AP exists
            # This is a good safety net in case the solver returns a slightly invalid solution due to tolerance
            S_set = set(S)
            valid = True
            for a in S:
                for b in S:
                    if a < b:
                        c = 2 * b - a
                        if c in S_set and c != a and c != b:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid and len(S) > 0:
                return S
            else:
                # If invalid, try to repair or fallback to a known good set (e.g., Moser set)
                # Moser set: numbers with only 0,1 in base 3
                def is_moser(x):
                    while x > 0:
                        if x % 3 == 2:
                            return False
                        x //= 3
                    return True
                moser = sorted([x for x in range(1, n + 1) if is_moser(x)])
                return moser
        else:
            # Fallback if solver fails
            def is_moser(x):
                while x > 0:
                    if x % 3 == 2:
                        return False
                    x //= 3
                return True
            moser = sorted([x for x in range(1, n + 1) if is_moser(x)])
            return moser
            
    except Exception as e:
        # Fallback on any error
        def is_moser(x):
            while x > 0:
                if x % 3 == 2:
                    return False
                x //= 3
            return True
        moser = sorted([x for x in range(1, n + 1) if is_moser(x)])
        return moser
