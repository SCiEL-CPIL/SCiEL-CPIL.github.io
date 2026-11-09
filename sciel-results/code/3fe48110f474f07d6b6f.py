# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    # Use the Möbius function for k = 1 to K
    K = 5000
    
    # Compute Möbius function using linear sieve
    mu = np.ones(K + 1, dtype=np.float64)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    
    # Get domain and values for k where mu(k) != 0
    nonzero_mask = mu[1:] != 0
    domain = np.where(nonzero_mask)[0].astype(np.int64) + 1
    values = mu[1:][nonzero_mask].astype(np.float64)
    
    return domain, values


# EVOLVE-BLOCK-END
