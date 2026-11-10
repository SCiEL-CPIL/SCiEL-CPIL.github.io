# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(n):
    """Compute Möbius function mu(1..n) via sieve."""
    mu = np.ones(n + 1, dtype=np.int8)
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
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


def _check_feasibility(domain, values, K, tol=1.0001):
    """Check M(x) = sum_k f(k)*floor(x/k) <= tol for all x in [1, 10K].
    Efficient: M(x) only changes at x = k for k in domain.
    """
    if len(domain) == 0:
        return True
    # Sort domain
    idx = np.argsort(domain)
    d = domain[idx]
    v = values[idx]
    
    max_x = 10 * K
    # M(x) for x in [d[i], d[i+1)-1] is constant = sum_{j: d[j] <= d[i]} v[j] * floor(d[i]/d[j])
    # Actually M(x) = sum_j v[j] * floor(x/d[j])
    # For x in [d[i], d[i+1]-1], floor(x/d[j]) is constant for each j
    # But computing this naively is O(m^2)
    
    # Better approach: M(x) changes at each x where x is a multiple of some d[j]
    # But that's too many points.
    
    # Alternative: Since M(x) = sum_j v[j] * floor(x/d[j]), and we need to check all x in [1, 10K],
    # we can evaluate M at strategic points. M(x) is a step function that increases at multiples.
    # The maximum of M(x) over [1, 10K] occurs at x = 10K or at points just before drops.
    # Actually M(x) is non-decreasing if all v[j] >= 0, but here v can be negative.
    
    # Let's just evaluate M(x) at x = 1, 2, ..., 10K using an efficient method.
    # M(x) = sum_j v[j] * floor(x/d[j])
    # For each j, floor(x/d[j]) changes at x = d[j], 2*d[j], 3*d[j], ...
    
    # Use difference array approach:
    # M(x+1) - M(x) = sum_j v[j] * [1 if (x+1) % d[j] == 0 else 0]
    # = sum over j where d[j] divides (x+1) of v[j]
    
    # Precompute: for each d in domain, add v to positions that are multiples of d
    # diff[x] = sum_{j: d[j] | x} v[j] for x = 1, 2, ..., 10K
    # Then M(x) = M(x-1) + diff[x], with M(0) = 0
    
    max_x = 10 * K
    diff = np.zeros(max_x + 1, dtype=np.float64)
    for j in range(len(d)):
        dv = d[j]
        vv = v[j]
        if dv <= 0:
            continue
        # Add vv to diff at positions dv, 2*dv, 3*dv, ... up to max_x
        for pos in range(dv, max_x + 1, dv):
            diff[pos] += vv
    
    # M(x) = cumsum(diff)[x]
    M = np.cumsum(diff)
    # Check max M(x) for x in [1, max_x]
    if len(M) > 1:
        max_M = np.max(M[1:])
        if max_M > tol:
            return False
    return True


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin
    
    # Compute Möbius function up to a reasonable bound
    # Start with a moderate N and increase if time permits
    N = 50000
    
    mu = _compute_mobius(N)
    
    # Domain: 1 to N
    domain = np.arange(1, N + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    
    # The Möbius function gives M(x) = 1 for all x >= 1 (in the infinite case).
    # For finite truncation, M(x) may exceed 1 slightly. Apply a safety factor.
    # Scale all values by a factor s < 1 to ensure M(x) <= 1.
    
    # First, check if unscaled is feasible
    K = N
    if _check_feasibility(domain, values, K):
        s = 1.0
    else:
        # Binary search for the maximum scaling factor s such that M(x) <= 1
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if _check_feasibility(domain, values * mid, K):
                lo = mid
            else:
                hi = mid
        s = lo * 0.999  # Additional safety margin
        if s <= 0:
            s = 0.5  # Fallback
    
    values = values * s
    
    # Check feasibility again
    if not _check_feasibility(domain, values, K):
        # Further reduce
        s = s * 0.5
        values = values * s
    
    return domain, values


# EVOLVE-BLOCK-END
