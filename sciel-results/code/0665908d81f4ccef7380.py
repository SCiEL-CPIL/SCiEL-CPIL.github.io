# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(N):
    """Compute Moebius function mu(k) for k = 1..N using linear sieve."""
    mu = np.zeros(N + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > N:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


def _verify_feasibility(domain, values, max_x):
    """
    Check M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for x in [1, max_x].
    Uses the fact that M(x) = sum_{n=1}^{x} g(n) where g(n) = sum_{k|n} f(k).
    """
    m = len(domain)
    if m == 0:
        return True

    # Compute g(n) = sum_{k|n} f(k) for n up to max_x
    g = np.zeros(max_x + 1, dtype=np.float64)
    for i in range(m):
        k = int(domain[i])
        v = float(values[i])
        # Add v to g(n) for all n that are multiples of k
        for n in range(k, max_x + 1, k):
            g[n] += v

    # M(x) = cumsum(g)[x]
    M = np.cumsum(g)
    # Check M(x) <= 1.0001 for x in [1, max_x]
    if np.any(M[1:max_x + 1] > 1.0001):
        return False
    return True


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = 600.0

    # Strategy: Use the Mobius function mu(k) for k = 1..K.
    # For the full Mobius function, M(x) = 1 for all x >= 1 (since sum_{d|n} mu(d) = 0 for n>1).
    # For a truncated version (k <= K), M(x) = 1 for x <= K, but for x > K, 
    # M(x) = sum_{k<=K} mu(k) * floor(x/k) which may deviate from 1.
    # 
    # The objective a = -sum_{k=1}^{K} mu(k) * log(k) / k approaches 1 as K -> infinity.
    # We want to find the largest K such that M(x) <= 1.0001 for all x in [1, 10K].

    best_domain = None
    best_values = None
    best_a = -np.inf

    # Try a range of K values
    K_candidates = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

    for K in K_candidates:
        if time.time() > deadline - 60:
            break

        max_x = 10 * K
        if max_x > 2000000:
            break

        mu = _compute_mobius(K)

        # domain: k where mu(k) != 0 (non-zero mu values matter)
        non_zero = np.where(mu[1:K + 1] != 0)[0] + 1
        if len(non_zero) == 0:
            continue

        domain = non_zero.astype(np.int64)
        values = mu[domain].astype(np.float64)

        # Verify feasibility
        if not _verify_feasibility(domain, values, max_x):
            continue

        # Compute objective a = -sum_k f(k) * log(k) / k
        log_k = np.log(domain.astype(np.float64))
        a = -np.sum(values * log_k / domain.astype(np.float64))

        if a > best_a:
            best_a = a
            best_domain = domain.copy()
            best_values = values.copy()

        # If a is very close to 1, we can stop
        if best_a > 0.999:
            break

    # Fallback: use a simple valid function
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values


# EVOLVE-BLOCK-END
