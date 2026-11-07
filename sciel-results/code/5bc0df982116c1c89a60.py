import math
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 256
    
    # Sieve to find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Precompute distinct prime factors for each number 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        x = i
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                prime_factors[i].add(p)
                while x % p == 0:
                    x //= p
        if x > 1:
            prime_factors[i].add(x)
    prime_factors[1] = set()
    
    # Try MILP first for exact solution
    best_set = None
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        num_primes = len(primes)
        # A matrix: num_primes x (N+1)
        # A[p_idx, n] = 1 if primes[p_idx] divides n
        A = np.zeros((num_primes, N + 1), dtype=int)
        for idx, p in enumerate(primes):
            for multiple in range(p, N + 1, p):
                A[idx, multiple] = 1
        
        # Objective: Maximize sum(n * x_n) -> Minimize sum(-n * x_n)
        c = np.zeros(N + 1)
        for n in range(1, N + 1):
            c[n] = -n
            
        # Constraints: For each prime, sum of x_n for multiples <= 1
        cons = LinearConstraint(A, lb=0, ub=1)
        
        # Bounds: 0 <= x_n <= 1, integer
        bounds = Bounds(0, 1)
        integrality = np.ones(N + 1)
        
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.success:
            selected = []
            for n in range(1, N + 1):
                if res.x[n] > 0.5:
                    selected.append(int(n))
            if selected:
                best_set = set(selected)
                best_sum = sum(best_set)
    except Exception:
        best_set = None
    
    # Fallback to local search if MILP failed or timed out
    if best_set is None:
        # Initial solution: 1 + for each prime, the largest power ≤ N
        best_set = {1}
        for p in primes:
            val = p
            while val * p <= N:
                val *= p
            best_set.add(val)
        
        # Compute the set of all primes currently used
        used_primes = set()
        for s in best_set:
            used_primes |= prime_factors[s]
        
        best_sum = sum(best_set)
        
        # Local search with multiple move types
        unchanged = 0
        while unchanged < 60 and time.time() < deadline - 3.0:
            improved = False
            
            # Move type 1: Add a single number
            for n in range(N, 0, -1):
                if n not in best_set and len(prime_factors[n] & used_primes) == 0:
                    best_set.add(n)
                    used_primes |= prime_factors[n]
                    best_sum += n
                    improved = True
                    unchanged = 0
                    break
            
            # Move type 2: 1-for-1 swap
            if not improved:
                for n_out in list(best_set):
                    if n_out == 1:
                        continue
                    freed = prime_factors[n_out]
                    remaining = used_primes - freed
                    for n_in in range(N, 0, -1):
                        if n_in not in best_set and len(prime_factors[n_in] & remaining) == 0:
                            new_sum = best_sum - n_out + n_in
                            if new_sum > best_sum:
                                best_set.remove(n_out)
                                best_set.add(n_in)
                                used_primes = remaining | prime_factors[n_in]
                                best_sum = new_sum
                                improved = True
                                unchanged = 0
                                break
                    if improved:
                        break
            
            # Move type 3: 2-for-1 swap
            if not improved:
                lst = [x for x in best_set if x != 1]
                for i in range(len(lst)):
                    for j in range(i + 1, len(lst)):
                        a, b = lst[i], lst[j]
                        freed = prime_factors[a] | prime_factors[b]
                        remaining = used_primes - freed
                        for n_in in range(N, 0, -1):
                            if n_in not in best_set and len(prime_factors[n_in] & remaining) == 0:
                                new_sum = best_sum - a - b + n_in
                                if new_sum > best_sum:
                                    best_set.remove(a)
                                    best_set.remove(b)
                                    best_set.add(n_in)
                                    used_primes = remaining | prime_factors[n_in]
                                    best_sum = new_sum
                                    improved = True
                                    unchanged = 0
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            # Move type 4: 1-for-2 swap
            if not improved:
                for n_out in list(best_set):
                    if n_out == 1:
                        continue
                    freed = prime_factors[n_out]
                    remaining = used_primes - freed
                    for n_in1 in range(N, 0, -1):
                        if n_in1 not in best_set and len(prime_factors[n_in1] & remaining) == 0:
                            rem2 = remaining | prime_factors[n_in1]
                            for n_in2 in range(n_in1 - 1, 0, -1):
                                if n_in2 not in best_set and n_in2 != n_in1 and len(prime_factors[n_in2] & rem2) == 0:
                                    new_sum = best_sum - n_out + n_in1 + n_in2
                                    if new_sum > best_sum:
                                        best_set.remove(n_out)
                                        best_set.add(n_in1)
                                        best_set.add(n_in2)
                                        used_primes = rem2 | prime_factors[n_in2]
                                        best_sum = new_sum
                                        improved = True
                                        unchanged = 0
                                        break
                            if improved:
                                break
                    if improved:
                        break
            
            if not improved:
                unchanged += 1
    
    # Convert to list of Python ints
    result = [int(x) for x in best_set]
    
    # Final validation
    assert len(set(result)) == len(result), "Duplicate values"
    assert all(isinstance(x, int) and 1 <= x <= N for x in result), "Invalid value"
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert math.gcd(result[i], result[j]) == 1, f"Non-coprime pair: {result[i]}, {result[j]}"
    
    return result
