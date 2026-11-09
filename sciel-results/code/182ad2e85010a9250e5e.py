# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy: Dirichlet convolution inversion using the exact Mobius function mu(k).
    We compute mu(k) for all k <= K (K=200000) and retain only square-free indices
    where mu(k) != 0. The values are set to mu(k).

    The classic identity sum_{k|x} mu(k) = 0 for x > 1 and = 1 for x = 1 implies
    M(x) = sum_k mu(k) * floor(x/k) = 1 for all x >= 1 (in the infinite case).
    For finite K, M(x) remains bounded and close to 1 for x in [1, 10K].

    The objective a = -sum_k f(k) * log(k) / k approximates 1 as K -> infinity
    since -sum_{k<=K} mu(k) log(k)/k -> 1.

    Returns:
        (domain, values): tuple of two numpy arrays
    """
    K = 200000

    # Compute Mobius function using sieve
    mu = np.ones(K + 1, dtype=np.int64)
    primes = []
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False

    # Standard sieve for primes and Mobius
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0

    # Retain only indices where mu(k) != 0 (square-free numbers)
    mask = mu != 0
    domain = np.where(mask)[1:]  # Exclude index 0, keep 1..K
    values = mu[domain].astype(np.float64)

    return domain, values


# EVOLVE-BLOCK-END
