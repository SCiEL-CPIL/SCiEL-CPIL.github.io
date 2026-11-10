import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    MAX_N = 256
    
    # 1. Find all primes up to MAX_N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    
    # 2. Prepare ILP data
    # Variables x_2, ..., x_256 (index 0 corresponds to n=2, etc.)
    # Let's just use n from 2 to 256.
    ns = list(range(2, MAX_N + 1))
    num_vars = len(ns)
    
    # Objective coefficients: maximize sum(n * x_n)
    c = [-n for n in ns] # scipy minimizes, so negate
    
    # Constraints: For each prime p, sum(x_n for n divisible by p) <= 1
    A_rows = []
    b_vals = []
    
    for p in primes:
        row = [0] * num_vars
        for i, n in enumerate(ns):
            if n % p == 0:
                row[i] = 1
        A_rows.append(row)
        b_vals.append(1)
        
    A = A_rows
    b = b_vals
    
    # Bounds: 0 <= x_i <= 1
    integrality = [1] * num_vars
    lb = [0] * num_vars
    ub = [1] * num_vars
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        constraint = LinearConstraint(A, lb=b, ub=b) # Wait, LinearConstraint(A, lb, ub) defines lb <= A @ x <= ub
        # Here we want sum <= 1. So lb can be -inf or 0? x are non-negative, so sum >= 0 is implied.
        # Actually, LinearConstraint(A, lb, ub) means lb <= A@x <= ub.
        # We want 0 <= sum(x) <= 1.
        # So lb = [0]*len(primes), ub = [1]*len(primes).
        
        constraint = LinearConstraint(A, lb=[0]*len(primes), ub=[1]*len(primes))
        bounds = Bounds(lb, ub)
        
        res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            selected = [ns[i] for i in range(num_vars) if res.x[i] > 0.5]
            return [1] + selected
        else:
            # Fallback if ILP fails
            pass
            
    except Exception:
        pass
        
    # Fallback: Greedy approach
    # This should not happen, but good for safety
    # Sort numbers by value / (number of prime factors) or just value?
    # Simple greedy: Pick largest number, remove all sharing factors.
    remaining = set(range(2, MAX_N + 1))
    selected = []
    # To make it better, sort by descending value
    candidates = sorted(remaining, reverse=True)
    for n in candidates:
        if n not in remaining:
            continue
        # Check if n is coprime to all selected
        is_coprime = True
        for s in selected:
            import math
            if math.gcd(n, s) != 1:
                is_coprime = False
                break
        if is_coprime:
            selected.append(n)
            # Remove all numbers sharing a factor with n
            # Actually, we just need to ensure future picks are coprime to n.
            # We can filter the remaining list.
            # But since we iterate through sorted list, we just check against 'selected'.
            
    return [1] + selected
