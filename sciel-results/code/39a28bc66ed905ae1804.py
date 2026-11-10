import numpy as np
import time
from scipy.optimize import linprog
import sys

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 550.0  # Safety margin
    
    # Try different K values
    # Start with small K to ensure we have a valid result quickly
    best_score = 0.0
    best_domain = None
    best_values = None
    
    # Fallback: Mobius function for small K
    def get_mobius(K):
        mu = np.ones(K + 1, dtype=np.int64)
        mu[0] = 0
        mu[1] = 1
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, K + 1):
            if is_prime[i]:
                mu[i:] *= -1
                for j in range(i*i, K + 1, i):
                    is_prime[j] = False
        # mu[k] is 0 if k has squared prime factor
        # The simple sieve above doesn't handle squared factors correctly for mu
        # Correct Mobius calculation:
        mu = np.ones(K + 1, dtype=np.int64)
        mu[0] = 0
        is_prime = np.ones(K + 1, dtype=bool)
        primes = []
        for i in range(2, K + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > K:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu
    
    # Heuristic baseline
    mu = get_mobius(100)
    domain = np.arange(1, 101)
    values = mu[1:101].astype(np.float64)
    values[0] = 0.0 # f(1) will be adjusted by evaluator
    
    # Calculate score for baseline
    # S = sum values/k
    S = np.sum(values / domain)
    values_adj = values.copy()
    values_adj[0] -= S
    # M(x) check
    max_M = 0.0
    for x in range(1, 1001):
        M_x = np.sum(values_adj * (x // domain))
        if M_x > max_M:
            max_M = M_x
    if max_M <= 1.0001:
        a = -np.sum(values_adj * np.log(domain) / domain)
        best_score = a
        best_domain = domain.copy()
        best_values = values.copy()
    else:
        # If Mobius fails (unlikely for K=100), try smaller K or seed
        pass

    # Try LP optimization for larger K
    for K in [200, 500, 1000]:
        if time.time() > deadline - 60:
            break
        
        # Prepare LP
        # Variables: f(2) ... f(K)
        n_vars = K - 1
        c = np.zeros(n_vars)
        for i in range(n_vars):
            k = i + 2
            c[i] = np.log(k) / k
        
        # Constraints: sum f(k) * {x/k} >= -1.0001
        # => -sum f(k) * {x/k} <= 1.0001
        n_constraints = 10 * K
        A_ub = np.zeros((n_constraints, n_vars))
        b_ub = np.full(n_constraints, 1.0001)
        
        for x in range(1, n_constraints + 1):
            row = x - 1
            for i in range(n_vars):
                k = i + 2
                if k <= x:
                    frac = (x / k) - int(x / k)
                    A_ub[row, i] = -frac
                else:
                    A_ub[row, i] = 0.0
        
        bounds = [(-10, 10)] * n_vars
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                f_vals = np.zeros(K + 1)
                f_vals[1] = 0.0
                f_vals[2:K+1] = res.x
                
                # Verify
                S = np.sum(f_vals[1:K+1] / np.arange(1, K+1))
                f_adj = f_vals.copy()
                f_adj[1] -= S
                
                # Check M(x)
                feasible = True
                for x in range(1, 10 * K + 1):
                    M_x = np.sum(f_adj[1:K+1] * (x // np.arange(1, K+1)))
                    if M_x > 1.0001:
                        feasible = False
                        break
                
                if feasible:
                    a = -np.sum(f_adj[1:K+1] * np.log(np.arange(1, K+1)) / np.arange(1, K+1))
                    if a > best_score:
                        best_score = a
                        best_domain = np.arange(1, K+1)
                        best_values = f_vals[1:K+1].copy()
                        
        except Exception as e:
            pass

    if best_domain is None:
        # Return seed-like result
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return best_domain, best_values
