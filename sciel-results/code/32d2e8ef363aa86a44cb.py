import time
import numpy as np
from math import gcd, isqrt
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 15s margin for safety

    MAX_N = 256

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(MAX_N)
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Pre-compute divisibility matrix for speed
    divisors = [[] for _ in range(MAX_N + 1)]
    for n in range(1, MAX_N + 1):
        for p in primes:
            if n % p == 0:
                divisors[n].append(p)

    # 1. Try ILP approach
    best_result = None
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix

        # Build constraint matrix
        # A is num_primes x MAX_N
        # We will build it as a list of tuples for sparse conversion
        rows = []
        cols = []
        data = []
        
        for n in range(1, MAX_N + 1):
            for p in divisors[n]:
                idx = prime_to_idx[p]
                rows.append(idx)
                cols.append(n - 1)
                data.append(1.0)
                
        A = csr_matrix((data, (rows, cols)), shape=(num_primes, MAX_N))

        # Objective: Minimize -sum(n * x_n)
        c = -np.arange(1, MAX_N + 1, dtype=float)

        # Constraints: A @ x <= 1
        constraints = LinearConstraint(A, lb=np.zeros(num_primes), ub=np.ones(num_primes))

        # Integrality: all integer
        integrality = np.ones(MAX_N)

        # Bounds: 0 <= x <= 1
        bounds = Bounds(lb=np.zeros(MAX_N), ub=np.ones(MAX_N))
        
        # Set a time limit for the solver to avoid hanging
        # Use 60 seconds, leaving 15s for rest
        options = {'time_limit': 60}

        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options=options)

        if res.status == 0:
            x = res.x
            selected = [int(i + 1) for i in range(MAX_N) if x[i] > 0.5]
            
            # Verify
            if len(selected) > 0:
                valid = True
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        if gcd(selected[i], selected[j]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_result = [int(n) for n in selected]
                    return best_result
    except Exception:
        pass

    # 2. Fallback: Improved Local Search
    # ... (keep existing fallback but maybe optimize slightly)
    
    S = []
    for n in range(256, 0, -1):
        if all(gcd(n, x) == 1 for x in S):
            S.append(n)
    
    best_set = set(S)
    best_sum = sum(S)
    
    # Local Search Optimization - multiple passes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = sorted(list(best_set))
        
        # Try to add any missing number
        for b in range(256, 1, -1):
            if b in best_set:
                continue
            if all(gcd(b, x) == 1 for x in best_set):
                best_set.add(b)
                best_sum += b
                improved = True
        
        # 1-for-1 swaps
        for a in S_list:
            if a == 1:
                continue
            for b in range(256, a, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set if x != a):
                    new_sum = best_sum - a + b
                    if new_sum > best_sum:
                        best_set.remove(a)
                        best_set.add(b)
                        best_sum = new_sum
                        improved = True
        
        # 2-for-1 swaps
        if time.time() < deadline:
            S_list = sorted(list(best_set))
            for i in range(len(S_list)):
                if time.time() >= deadline:
                    break
                a1 = S_list[i]
                if a1 == 1:
                    continue
                for j in range(i + 1, len(S_list)):
                    a2 = S_list[j]
                    if a2 == 1:
                        continue
                    for b in range(256, a1 + a2, -1):
                        if b in best_set or b == a1 or b == a2:
                            continue
                        if all(gcd(b, x) == 1 for x in best_set if x not in (a1, a2)):
                            best_set.remove(a1)
                            best_set.remove(a2)
                            best_set.add(b)
                            best_sum = best_sum - a1 - a2 + b
                            improved = True
                            break
    
    # Final polish
    if 1 not in best_set:
        if all(gcd(1, x) == 1 for x in best_set):
            best_set.add(1)
            best_sum += 1

    return [int(x) for x in sorted(best_set)]
