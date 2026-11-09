# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Mobius function mu(n) for n = 1..K using a sieve."""
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
    # Handle squares: mu[n] = 0 if n has a squared prime factor
    for p in primes:
        p2 = p * p
        if p2 > K:
            break
        for j in range(p2, K + 1, p2):
            mu[j] = 0.0
    return mu


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
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin before 600s limit

    # Start with a moderate K to ensure the evaluator (which scales with K) finishes in time.
    # K=100000 was too slow. K=10000 should be fast.
    K = 10000
    
    # Check if we have enough time to compute for larger K if needed, 
    # but for now, stick to K=10000 to guarantee execution.
    
    mu = _compute_mobius(K)

    # Get domain: all k where mu(k) != 0
    domain = np.array([k for k in range(1, K + 1) if mu[k] != 0.0], dtype=np.int64)
    values = mu[domain]

    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback to a very small seed-like result
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    return domain, values


# EVOLVE-BLOCK-END
