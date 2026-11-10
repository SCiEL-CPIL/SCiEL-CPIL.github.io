# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(N):
    """Compute Moebius function mu(k) for k = 1..N using linear sieve."""
    mu = np.ones(N + 1, dtype=np.int8)
    primes = []
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
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
    mu[1] = 1
    return mu


def _check_feasibility(domain, values, K):
    """Check if M(x) <= 1.0001 for all x in [1, 10K]."""
    max_x = 10 * K
    chunk_size = 5000
    max_m = -np.inf
    m = len(domain)
    for start in range(1, max_x + 1, chunk_size):
        end = min(start + chunk_size, max_x + 1)
        x_vals = np.arange(start, end, dtype=np.int64)
        # Compute floor(x/k) for all x in chunk and all k in domain
        # Shape: (chunk_size, m)
        floors = (x_vals[:, np.newaxis] // domain[np.newaxis, :])
        M_vals = floors @ values
        chunk_max = np.max(M_vals)
        max_m = max(max_m, chunk_max)
        if max_m > 1.0001:
            return False, max_m
    return True, max_m


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = start_time + 550

    fallback_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    fallback_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    # Try K values from large to small, return first feasible
    for K in [100000, 50000, 20000, 10000, 5000, 2000, 1000]:
        if time.time() > deadline - 60:
            break

        mu = _compute_mobius(K)
        mask = mu != 0
        domain = np.where(mask)[0]
        domain = domain[domain > 0]
        values = mu[domain].astype(np.float64)

        if len(domain) == 0:
            continue

        feasible, max_M = _check_feasibility(domain, values, K)

        if feasible:
            return domain, values

    return fallback_domain, fallback_values


# EVOLVE-BLOCK-END
