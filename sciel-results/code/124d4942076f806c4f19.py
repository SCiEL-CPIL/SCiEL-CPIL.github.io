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
    # Maximum key allowed by the task contract
    K_max = 200000

    # Compute the Möbius function mu(k) for k in [1, K_max] using a linear sieve.
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

    # Domain: all k in [1, K_max] where mu(k) != 0 (square-free integers)
    # Values: mu(k) for those k
    # Note: The evaluator will clip values to [-10, 10], which doesn't affect mu(k) in {-1, 0, 1}
    # The evaluator will adjust f(1) so that sum_k f(k)/k = 0.
    # We include k=1 in the domain with mu(1)=1, so f(1) will be adjusted.

    mask = mu != 0
    domain = np.where(mask)[0]
    # Ensure domain starts at 1 (index 0 in the array corresponds to k=0, but we want k>=1)
    # Actually, mu is defined for indices 0..K_max, with mu[0] unused.
    # np.where(mask)[0] gives indices where mask is True. Since mu[0]=0, index 0 is excluded.
    # The indices are 1..K_max, which are the actual keys.
    
    # Filter to only positive integers (domain keys should be >= 1)
    domain = domain[domain >= 1]
    values = mu[domain].astype(np.float64)

    return domain, values


# EVOLVE-BLOCK-END
