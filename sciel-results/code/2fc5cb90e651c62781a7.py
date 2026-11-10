# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 600.0
    
    # Maximum key allowed by the task contract
    K = 200000
    
    # Compute the Mobius function mu(k) for k = 1..K
    # mu(1) = 1
    # mu(k) = 0 if k has a squared prime factor
    # mu(k) = (-1)^n if k is a product of n distinct primes
    
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    # Linear sieve for Mobius function
    mu[0] = 0
    mu[1] = 1
    
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

    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:K + 1].astype(np.float64)
    
    return domain, values

# EVOLVE-BLOCK-END
