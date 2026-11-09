import numpy as np


def prime_number_theorem_partial_function():
    K = 5000
    
    # Compute Möbius function using linear sieve
    mu = np.zeros(K + 1, dtype=np.float64)
    mu[1] = 1.0
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1.0
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0.0
                break
            else:
                mu[i * p] = -mu[i]
    
    # Build domain and values for k where mu(k) != 0
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[1:][mask]
    
    return domain, values
