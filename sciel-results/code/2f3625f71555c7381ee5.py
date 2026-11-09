import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 500  # Safety margin
    
    # Max K to consider
    K_max = 50000
    
    # Compute Mobius function
    mu = np.ones(K_max + 1, dtype=np.int8)
    is_prime = np.ones(K_max + 1, dtype=bool)
    primes = []
    for i in range(2, K_max + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
            for p in primes:
                if i * p > K_max:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
    
    # Compute objective for each K
    # a_K = - sum_{k=1}^K mu(k) log(k)/k
    # We want to maximize a_K
    logs = np.log(np.arange(1, K_max + 1))
    terms = mu * logs / np.arange(1, K_max + 1)
    cumsum = np.cumsum(terms)
    objectives = -cumsum
    
    # Find K that maximizes objective
    best_K = np.argmax(objectives) + 1
    
    # Check time
    if time.time() > deadline:
        # Fallback to small K
        best_K = 100
        
    domain = np.arange(1, best_K + 1, dtype=np.int64)
    values = mu[1:best_K + 1].astype(np.float64)
    
    return domain, values
