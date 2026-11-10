# EVOLVE-BLOCK-START
import numpy as np
import math


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy:
        Compute the exact Möbius function mu(k) for all k in [1, K_max] using a linear sieve.
        The domain is the set of all square-free integers k (where mu(k) != 0) up to K_max.
        The values are mu(k).
        
        This construction satisfies the constraint M(x) = sum_k f(k) * floor(x/k) <= 1
        with equality for all x (a classic identity involving the Möbius function),
        and maximizes the objective a = -sum_k f(k) * log(k) / k by including
        the full range of contributing terms up to K_max.
    """
    K_max = 200000
    
    # Linear sieve to compute Möbius function
    mu = np.ones(K_max + 1, dtype=np.int8)
    primes = []
    is_prime = np.ones(K_max + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # Initialize mu[1] = 1
    mu[1] = 1
    
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
    
    # Collect domain and values where mu(k) != 0
    mask = mu != 0
    domain = np.nonzero(mask)[0]
    domain = domain[domain > 0]  # Only positive integers
    values = mu[domain].astype(np.float64)
    
    return domain, values


# EVOLVE-BLOCK-END
