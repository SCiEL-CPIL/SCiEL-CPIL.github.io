# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
        * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    K_max = 200000
    
    # Linear sieve to compute the Mobius function mu(k) for k in [1, K_max]
    # mu(1) = 1
    # mu(n) = 0 if n has a squared prime factor
    # mu(n) = (-1)^k if n is a product of k distinct primes
    
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
    
    # Extract domain and values where mu(k) != 0
    # These are the square-free integers
    nonzero_indices = np.nonzero(mu[1:])[0] + 1  # +1 because we started from index 1
    domain = nonzero_indices.astype(np.int64)
    values = mu[1:][nonzero_indices - 1].astype(np.float64)
    
    return domain, values


# EVOLVE-BLOCK-END
