# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy: Use the exact Mobius function mu(k) for k up to K=200000, retaining only
    square-free indices where mu(k) != 0. This directly approximates the infinite identity
    sum mu(k)/k = 0 and M(x) = 1 for all x, maximizing the objective a = -sum mu(k) log(k)/k.
    """
    K = 200000

    # Compute the Mobius function mu(k) for k = 1..K using a sieve.
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
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

    # Retain only indices where mu(k) != 0 (square-free integers)
    mask = mu != 0
    domain = np.nonzero(mask)[0]
    domain = domain[domain > 0]  # positive integers only
    values = mu[domain].astype(np.float64)

    return domain, values


# EVOLVE-BLOCK-END
