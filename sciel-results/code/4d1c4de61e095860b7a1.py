import math
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 80.0
    N = 256

    # 1. Preprocessing
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
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

    best_result = None
    
    # 2. Attempt Exact Solution via MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        num_vars = N
        num_primes = len(primes)
        
        c = np.zeros(num_vars, dtype=np.float64)
        for n in range(1, N + 1):
            c[n - 1] = -float(n)
            
        A = np.zeros((num_primes, num_vars), dtype=np.int8)
        for p_idx, p in enumerate(primes):
            for n in range(1, N + 1):
                if n % p == 0:
                    A[p_idx, n - 1] = 1
                    
        cons = LinearConstraint(A, lb=0, ub=1)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(num_vars, dtype=np.int8)
        
        time_left = max(1.0, deadline - time.time() - 10.0)
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds, 
                   options={'time_limit': time_left})
        
        if res.x is not None:
            sol_set = set()
            for i, val in enumerate(res.x):
                if val > 0.5:
                    sol_set.add(i + 1)
            
            if 1 not in sol_set:
                sol_set.add(1)
                
            # Validate
            lst = list(sol_set)
            valid = True
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    if math.gcd(lst[i], lst[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_result = lst
    except Exception:
        pass

    # 3. Local Search Polish
    if best_result is None:
        # Greedy init
        best_set = {1}
        used_primes = set()
        for n in range(N, 1, -1):
            if not (prime_factors[n] & used_primes):
                best_set.add(n)
                used_primes |= prime_factors[n]
    else:
        best_set = set(best_result)
        
    best_sum = sum(best_set)
    used_primes = set()
    for s in best_set:
        used_primes |= prime_factors[s]
        
    pf_frozen = [frozenset(prime_factors[i]) for i in range(N + 1)]
    
    unchanged = 0
    max_unchanged = 50
    while unchanged < max_unchanged and time.time() < deadline - 5.0:
        improved = False
        
        # 1-for-1 swap
        best_gain = 0
        best_move = None
        for n_out in list(best_set):
            if n_out == 1: continue
            freed = pf_frozen[n_out]
            remaining = used_primes - freed
            for n_in in range(N, 1, -1):
                if n_in not in best_set and len(pf_frozen[n_in] & remaining) == 0:
                    gain = n_in - n_out
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (n_out, n_in)
                    if n_in <= n_out: break
        if best_move:
            n_out, n_in = best_move
            best_set.remove(n_out)
            best_set.add(n_in)
            used_primes = (used_primes - pf_frozen[n_out]) | pf_frozen[n_in]
            best_sum += best_gain
            improved = True

        # Add single
        if not improved:
            for n in range(N, 1, -1):
                if n not in best_set and len(pf_frozen[n] & used_primes) == 0:
                    best_set.add(n)
                    used_primes |= pf_frozen[n]
                    best_sum += n
                    improved = True
                    break

        if not improved:
            unchanged += 1
        else:
            unchanged = 0

    if best_result is None:
        best_result = list(best_set)
    else:
        if best_sum > sum(best_result):
            best_result = list(best_set)

    return [int(x) for x in best_result]
