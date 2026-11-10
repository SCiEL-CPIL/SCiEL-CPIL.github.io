# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, leaving margin before 90s kill
    
    MAX_N = 256
    
    # Generate primes up to MAX_N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    
    # For each number 1..256, find its prime factors
    prime_factors = {}
    for n in range(1, MAX_N + 1):
        if n == 1:
            prime_factors[n] = []
        else:
            factors = []
            temp = n
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    factors.append(p)
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                factors.append(temp)
            prime_factors[n] = factors
    
    def is_pairwise_coprime(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True
    
    def local_search_polish(selected, time_limit=5.0):
        """Try to improve the solution by swapping or adding elements."""
        selected_set = set(selected)
        current_sum = sum(selected)
        best_sum = current_sum
        best_set = set(selected)
        
        # Try adding single elements
        for n in range(MAX_N, 0, -1):
            if n not in selected_set and all(gcd(n, x) == 1 for x in selected_set):
                new_sum = current_sum + n
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_set = selected_set | {n}
        
        # Try swapping: remove one, add one
        for remove in list(selected_set):
            if time.time() - start_time > deadline - 10:
                break
            remaining = selected_set - {remove}
            for n in range(MAX_N, 0, -1):
                if n not in selected_set and all(gcd(n, x) == 1 for x in remaining):
                    new_sum = current_sum - remove + n
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_set = remaining | {n}
        
        # Try removing one, adding two (composite pairing)
        if time.time() - start_time < deadline - 15:
            for remove in list(selected_set):
                if time.time() - start_time > deadline - 10:
                    break
                remaining = selected_set - {remove}
                for n1 in range(MAX_N, 0, -1):
                    if n1 not in selected_set and all(gcd(n1, x) == 1 for x in remaining):
                        for n2 in range(MAX_N, 0, -1):
                            if (n2 not in selected_set and n2 != n1 and 
                                all(gcd(n2, x) == 1 for x in remaining) and 
                                gcd(n1, n2) == 1):
                                new_sum = current_sum - remove + n1 + n2
                                if new_sum > best_sum:
                                    best_sum = new_sum
                                    best_set = remaining | {n1, n2}
        
        return list(best_set)
    
    # Try ILP via scipy.optimize.milp
    selected = None
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Build constraint matrix: one row per prime, one column for numbers 1..256
        n_primes = len(primes)
        n_vars = MAX_N
        
        # Build A matrix
        A = np.zeros((n_primes, n_vars), dtype=np.int8)
        for p_idx, p in enumerate(primes):
            for multiple in range(p, MAX_N + 1, p):
                A[p_idx, multiple - 1] = 1
        
        # Objective: maximize sum(n * x_n) => minimize -sum(n * x_n)
        c = -np.arange(1, MAX_N + 1, dtype=np.float64)
        
        # Constraints: A @ x <= 1 for all primes
        constraint = LinearConstraint(A, lb=0, ub=1)
        
        # Bounds: 0 <= x_n <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: all variables are integer (binary)
        integrality = np.ones(n_vars)
        
        # Try with different solver options
        for solver in ['highs', 'cplex', 'gurobi']:
            if time.time() - start_time > deadline - 15:
                break
            try:
                result = milp(c=c, constraints=constraint, bounds=bounds, 
                             integrality=integrality, options={"solver": solver})
                if result.success:
                    candidate = [i + 1 for i in range(n_vars) if result.x[i] > 0.5]
                    if is_pairwise_coprime(candidate):
                        if selected is None or sum(candidate) > sum(selected):
                            selected = candidate
            except Exception:
                continue
    except Exception:
        pass
    
    # If no ILP solution, use greedy
    if selected is None:
        S = set()
        for n in range(MAX_N, 0, -1):
            if all(gcd(n, x) == 1 for x in S):
                S.add(n)
        selected = list(S)
    
    # Apply local search polish
    if time.time() - start_time < deadline - 5:
        selected = local_search_polish(selected)
    
    # Verify and return
    selected = sorted(set(selected))
    if not is_pairwise_coprime(selected):
        # Fallback to safe greedy
        S = [1]
        for n in range(MAX_N, 1, -1):
            if all(gcd(n, x) == 1 for x in S):
                S.append(n)
        return [int(x) for x in sorted(S)]
    
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
