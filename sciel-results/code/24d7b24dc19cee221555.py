# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 500  # Safety margin

    def compute_mobius(K):
        mu = np.ones(K + 1, dtype=np.int64)
        mu[0] = 0
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        primes = []
        for i in range(2, K + 1):
            if is_prime[i]:
                primes.append(i)
                for j in range(i * i, K + 1, i):
                    is_prime[j] = False
                    mu[j] *= -1
            for p in primes:
                if i * p > K:
                    break
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    def evaluate_and_check(K):
        if K < 1:
            return -np.inf, None, None
        
        mu = compute_mobius(K)
        ks = np.arange(1, K + 1)
        fs = mu[1:K+1].astype(np.float64)
        
        # Calculate f(1) to satisfy sum f(k)/k = 0
        if K > 1:
            sum_fk_over_k = np.sum(fs[1:] / ks[1:])
            fs[0] = -sum_fk_over_k
        else:
            fs[0] = 0.0
            
        # Clip values
        fs = np.clip(fs, -10, 10)
        
        # Check Feasibility: M(x) <= 1.0001 for x in [1, 10K]
        max_x = 10 * K
        # Vectorized computation of M(x)
        # To save memory, we can do it in chunks if K is large, but K<=1000 is fine
        k_vec = np.arange(1, K + 1)
        x_vec = np.arange(1, max_x + 1)[:, None]
        floors = np.floor(x_vec / k_vec[None, :])
        M_vals = np.sum(fs[None, :] * floors, axis=1)
        
        max_M = np.max(M_vals)
        if max_M > 1.0001:
            return -np.inf, None, None
            
        # Calculate Objective
        # a = - sum f(k) log(k) / k
        # log(1) = 0, so f(1) term is 0
        obj = -np.sum(fs * np.log(ks) / ks)
        
        return obj, ks, fs

    best_obj = -np.inf
    best_domain = None
    best_values = None
    
    # Try various K
    # We want to find if any K gives obj > 1.0
    # Start with larger K, but check time
    for K in [1000, 800, 600, 400, 200, 100, 50, 20, 10, 5, 2, 1]:
        if time.time() > deadline:
            break
        obj, domain, values = evaluate_and_check(K)
        if obj > best_obj:
            best_obj = obj
            best_domain = domain
            best_values = values
            
    if best_domain is not None:
        return best_domain, best_values
        
    # Fallback
    return np.array([1], dtype=np.int64), np.array([0.0], dtype=np.float64)

# EVOLVE-BLOCK-END
