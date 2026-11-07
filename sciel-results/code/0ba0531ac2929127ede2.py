"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    n = 256
    num_v = n - 1  # 255 vertices, indexed 0..254, vertex i = number i+2
    weights = [i + 2 for i in range(num_v)]
    
    # Build conflict matrix: conflict[i][j] = 1 if gcd(i+2, j+2) > 1
    # For MILP: x_i + x_j <= 1 for all conflicting pairs (i != j)
    # We only need to add constraints for non-coprime pairs
    
    # Precompute conflict pairs
    conflict_pairs = []
    for i in range(num_v):
        val_i = i + 2
        for j in range(i + 1, num_v):
            val_j = j + 2
            if gcd(val_i, val_j) > 1:
                conflict_pairs.append((i, j))
    
    # Build constraint matrix for MILP
    # Each row: x_i + x_j <= 1 for a conflict pair
    m_constraints = len(conflict_pairs)
    A = np.zeros((m_constraints, num_v), dtype=np.float64)
    for k, (i, j) in enumerate(conflict_pairs):
        A[k, i] = 1.0
        A[k, j] = 1.0
    
    # Objective: maximize sum of weights * x
    c = np.array(weights, dtype=np.float64)
    
    # Bounds: 0 <= x_i <= 1
    lb = np.zeros(num_v, dtype=np.float64)
    ub = np.ones(num_v, dtype=np.float64)
    
    # Integrality: all variables integer (0 or 1)
    integrality = np.ones(num_v, dtype=np.int8)
    
    # Constraints: A @ x <= ones
    constraint = LinearConstraint(A, lb=np.zeros(m_constraints), ub=np.ones(m_constraints))
    bounds = Bounds(lb, ub)
    
    # Solve MILP
    try:
        res = milp(c=-c, constraints=constraint, integrality=integrality, bounds=bounds,
                   options={"time_limit": 50})
        
        if res.status == 0 and res.x is not None:
            # Extract solution
            selected = [int(i) for i in range(num_v) if res.x[i] > 0.5]
            result = [1]  # Always include 1
            for idx in selected:
                result.append(idx + 2)
            return [int(x) for x in result]
        else:
            # Fallback: greedy solution
            pass
    except Exception:
        pass
    
    # Fallback: greedy approach if MILP fails
    selected = []
    used_primes = set()
    
    def get_prime_factors(x):
        factors = set()
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return factors
    
    # Try numbers from largest to smallest
    for val in range(256, 1, -1):
        pf = get_prime_factors(val)
        if not pf & used_primes:
            selected.append(val)
            used_primes |= pf
    
    result = [1] + sorted(selected, reverse=True)
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
