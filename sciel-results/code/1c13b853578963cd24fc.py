import time
import math
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    # Safety margin: 90s limit, use 60s for solving to be safe
    deadline = start_time + 60.0
    
    n = 256
    nums = list(range(1, n + 1))
    
    # Identify conflicting pairs (gcd > 1)
    # We can optimize this by only checking pairs where gcd > 1
    # Using a sieve-like approach to find prime factors might be faster, 
    # but N=256 is small enough for direct GCD checks.
    conflicts = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if math.gcd(i, j) > 1:
                conflicts.append((i - 1, j - 1))
                
    # If no conflicts (impossible for n>1), return all
    if not conflicts:
        return nums

    # Objective: Maximize sum of selected numbers
    # Minimize -sum
    c = np.array([-float(i) for i in nums])
    
    # Constraints: x_i + x_j <= 1 for each conflict (i, j)
    # Build sparse matrix A
    rows = []
    cols = []
    data = []
    
    for idx, (i, j) in enumerate(conflicts):
        rows.append(idx)
        cols.append(i)
        data.append(1.0)
        
        rows.append(idx)
        cols.append(j)
        data.append(1.0)
        
    A = sp.csr_matrix((data, (rows, cols)), shape=(len(conflicts), n))
    
    # Linear Constraint: A @ x <= 1
    constraint = LinearConstraint(A, ub=1.0)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0.0, ub=1.0)
    
    # Integrality: All variables must be integer (0 or 1)
    integrality = np.ones(n, dtype=int) # 1 means integer
    
    try:
        # Solve ILP
        res = milp(
            c=c,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds
        )
        
        # Check time
        if time.time() > deadline:
             raise TimeoutError()
             
        if res.success:
            selected = []
            for i in range(n):
                if res.x[i] > 0.5:
                    selected.append(nums[i])
            # Ensure non-empty (1 is always coprime, so it should be selected)
            if not selected:
                selected = [1]
            return selected
        else:
            # If solver fails, fallback to heuristic
            pass
            
    except Exception:
        pass

    # Fallback: Heuristic - Primes + 1
    # This is guaranteed valid and provides a decent baseline
    from math import isqrt
    primes = [a for a in range(2, n + 1) if all(a % d for d in range(2, isqrt(a) + 1))]
    return [1] + primes
