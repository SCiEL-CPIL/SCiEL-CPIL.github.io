# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(n):
    """Compute Mobius function mu(1..n) using linear sieve."""
    mu = np.ones(n + 1, dtype=np.int8)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 600s limit

    # Use the Mobius function which gives M(x)=1 before adjustment.
    # After f(1) adjustment by -S, M(x) = 1 - S*x which is always <= 1.
    # The objective a = -sum_{k>=2} mu(k)*log(k)/k approaches 1 as K->infinity.
    
    # Use maximum feasible K to get closest to theoretical optimum
    K = 200000
    
    # Time check before expensive computation
    if time.time() - start_time > 60:
        K = 100000
    
    mu = _compute_mobius(K)
    
    # Time check after Mobius computation
    if time.time() - start_time > 300:
        # Reduce to a smaller set if running out of time
        K = 50000
        mu = _compute_mobius(K)
    
    # Select domain: all k where mu(k) != 0 (square-free numbers)
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1  # 1-indexed
    values = mu[1:][mask].astype(np.float64)
    
    # Clip values to [-10, 10] (already in {-1, 0, 1})
    values = np.clip(values, -10.0, 10.0)
    
    # Compute S = sum f(k)/k over clipped values
    S = np.sum(values / domain.astype(np.float64))
    
    # Adjust f(1): f(1) = mu(1) - S = 1 - S
    if domain[0] == 1:
        idx = np.where(domain == 1)[0][0]
        values[idx] -= S
    else:
        values = np.concatenate([[ -S ], values])
        domain = np.concatenate([[1], domain])
    
    # Feasibility check: M(x) = sum_k f(k)*floor(x/k) must be <= 1.0001
    # For x=1: M(1) = f(1) = 1-S <= 1.0001 (always true since S>=0)
    # For x>=2: M(x) = 1 - S*x <= 1 (since S>0)
    # So feasibility is guaranteed. Quick spot check:
    max_K = int(np.max(domain))
    for x in [1, 2, max_K]:
        M_x = np.sum(values * (x // domain))
        if M_x > 1.0001:
            # Shouldn't happen, but fallback
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
    
    return domain, values


# EVOLVE-BLOCK-END
