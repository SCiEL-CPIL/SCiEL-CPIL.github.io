# EVOLVE-BLOCK-START
import numpy as np
import math


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy:
      1. Compute the exact Möbius function mu(k) for all k in [1, K_max] via a linear sieve.
      2. Set the domain to all k in [1, K_max] where mu(k) != 0 (square-free integers).
      3. Set values to mu(k) for those k.
      4. The evaluator clips values to [-10, 10] (mu(k) is already in {-1, 0, 1}, so no change).
      5. The evaluator subtracts S = sum_k f(k)/k from f(1) to enforce sum_k f(k)/k = 0.
         For the full Möbius function, sum_{k=1}^{K} mu(k)/k is small (O(1/K) by PNT-related bounds),
         so the adjustment to f(1) is tiny and M(x) remains <= 1 + epsilon.
      6. The objective a = -sum_k f(k) * log(k) / k is maximized by including many terms.
         For the full Möbius function up to K, a approximates 1.0 (the theoretical optimum).

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    K_max = 200000

    # Linear sieve to compute mu(k) for k = 1..K_max
    mu = np.zeros(K_max + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K_max + 1, dtype=bool)

    for i in range(2, K_max + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K_max:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # Domain: all k in [1, K_max] where mu(k) != 0
    mask = mu != 0
    domain = np.nonzero(mask)[0].astype(np.int64)
    values = mu[mask].astype(np.float64)

    return domain, values


# EVOLVE-BLOCK-END
