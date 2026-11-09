# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy: Use the exact Mobius function mu(k) for all square-free k up to K=200000,
    with f(k) = mu(k). This leverages the classic identity sum_{k|x} mu(k) = 0 for x > 1,
    which implies M(x) = 1 for all x, making the construction feasible (M(x) <= 1.0001).
    The objective a = -sum f(k) log(k)/k = -sum mu(k) log(k)/k approximates 1 from below
    as K grows, yielding a score slightly above 1.0 compared to the seed's 5-point heuristic.
    """
    K = 200000

    # Compute Mobius function mu(k) for k = 1 to K using a linear sieve
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)

    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # Select square-free indices (mu[k] != 0) and set f(k) = mu(k)
    domain_mask = mu[1:] != 0
    domain = np.where(domain_mask)[0] + 1
    values = mu[domain].astype(np.float64)

    return domain, values


# EVOLVE-BLOCK-END
