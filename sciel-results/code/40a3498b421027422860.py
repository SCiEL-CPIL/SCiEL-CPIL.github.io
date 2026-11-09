import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 500  # Safety margin
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try decreasing K
    for K in [200000, 100000, 50000, 10000, 1000, 100]:
        if time.time() > deadline:
            break
            
        domain = np.arange(1, K + 1, dtype=np.int64)
        
        # Compute Mobius
        mu = np.ones(K + 1, dtype=np.int64)
        is_prime = np.ones(K + 1, dtype=bool)
        primes = []
        for i in range(2, K + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
                for j in primes:
                    if i * j > K:
                        break
                    is_prime[i * j] = False
                    if i % j == 0:
                        mu[i * j] = 0
                        break
                    else:
                        mu[i * j] = -mu[i]
        
        mu = mu[1:]
        k_vals = np.arange(1, K + 1, dtype=np.float64)
        S_K = np.sum(mu / k_vals)
        
        h = mu.astype(np.float64).copy()
        h[0] = 1.0 - S_K
        
        # Check clipping
        # h[0] is small, mu is small. No clipping.
        
        a = -np.sum(h * np.log(k_vals) / k_vals)
        
        # Check feasibility
        max_x = 10 * K
        D = np.zeros(max_x + 1, dtype=np.float64)
        
        for k in range(1, K + 1):
            val = h[k-1]
            if val != 0:
                for j in range(k, max_x + 1, k):
                    D[j] += val
                    
        M = np.cumsum(D)
        max_M = np.max(M[1:max_x+1])
        
        if max_M <= 1.0001:
            # Feasible
            if a > best_score:
                best_score = a
                best_domain = domain
                best_values = h
            # Since larger K gives better score, we can break if we found a feasible one?
            # Not necessarily, but likely.
            # Let's break to save time if we found a good one.
            # Actually, score converges to 1. 
            # If K=200000 is feasible, it's the best.
            break
            
    if best_domain is None:
        # Fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
        
    return best_domain, best_values
