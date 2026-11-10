import time
from math import isqrt, gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
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
    
    # Precompute prime factors for each number 2..256
    prime_factors = {}
    for n in range(2, MAX_N + 1):
        pf = set()
        m = n
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                pf.add(p)
                while m % p == 0:
                    m //= p
        if m > 1:
            pf.add(m)
        prime_factors[n] = pf
    
    # 2. Prepare ILP data
    ns = list(range(2, MAX_N + 1))
    num_vars = len(ns)
    
    # Objective coefficients: maximize sum(n * x_n), scipy minimizes so negate
    c = np.array([-n for n in ns], dtype=float)
    
    # Build constraint matrix using NumPy for speed
    # For each prime p, constraint: sum(x_n for n divisible by p) <= 1
    A = np.zeros((len(primes), num_vars), dtype=np.float64)
    for pi, p in enumerate(primes):
        for i, n in enumerate(ns):
            if n % p == 0:
                A[pi, i] = 1.0
    
    b = np.ones(len(primes), dtype=np.float64)
    
    # Bounds: 0 <= x_i <= 1
    integrality = np.ones(num_vars, dtype=int)
    lb = np.zeros(num_vars, dtype=float)
    ub = np.ones(num_vars, dtype=float)
    
    selected = None
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        constraint = LinearConstraint(A, lb=np.zeros(len(primes), dtype=float), ub=b)
        bounds = Bounds(lb, ub)
        
        res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status == 0:  # Optimal
            selected = [ns[i] for i in range(num_vars) if res.x[i] > 0.5]
        else:
            # Fallback to greedy if ILP doesn't converge
            pass
            
    except Exception:
        pass
    
    if selected is None:
        # Fallback: Greedy approach
        selected = []
        candidates = sorted(range(2, MAX_N + 1), reverse=True)
        for n in candidates:
            is_coprime = True
            for s in selected:
                if gcd(n, s) != 1:
                    is_coprime = False
                    break
            if is_coprime:
                selected.append(n)
    
    # Local search refinement: try swapping elements to improve sum
    # Only run if we have time
    if time.time() < deadline - 10.0:
        selected_set = set(selected)
        sum_val = sum(selected)
        
        # Try each pair: remove one selected, add one unselected if coprime to all remaining
        unselected = [n for n in range(2, MAX_N + 1) if n not in selected_set]
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for idx, rem in enumerate(selected):
                if time.time() > deadline - 5.0:
                    break
                remaining_set = selected_set - {rem}
                for add in unselected:
                    if add == rem:
                        continue
                    # Check if add is coprime to all in remaining_set
                    can_add = True
                    for s in remaining_set:
                        if gcd(add, s) != 1:
                            can_add = False
                            break
                    if can_add:
                        new_sum = sum_val - rem + add
                        if new_sum > sum_val:
                            # Accept the swap
                            selected[idx] = add
                            selected_set.remove(rem)
                            selected_set.add(add)
                            unselected.remove(rem)
                            unselected.append(add)
                            sum_val = new_sum
                            improved = True
                            break
    
    result = [1] + selected
    # Ensure all are Python ints
    result = [int(x) for x in result]
    return result
