"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict

def _sigma_sieve(n):
    """Compute sigma(k) for k=1..n using a numpy-accelerated sieve."""
    sigma = np.zeros(n + 1, dtype=np.int64)
    for d in range(1, n + 1):
        sigma[d::d] += d
    return sigma

def _sigma_exact(n):
    """Exact sigma(n) via trial division (with isqrt cutoff)."""
    if n <= 0:
        return 0
    s = 1
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            term = 1
            pk = 1
            while x % p == 0:
                x //= p
                pk *= p
                term += pk
            s *= term
        p += 1 if p == 2 else 2
    if x > 1:
        s *= (1 + x)
    return s

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    # Start with the known 3-way collision as fallback
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # Search for collisions using numpy-accelerated sieve
    # k * sigma(k) <= 10^18
    # For practical purposes, we can go up to ~20M with numpy
    
    for limit in [5_000_000, 10_000_000, 20_000_000, 40_000_000]:
        if time.time() > deadline - 30:
            break
        try:
            sigma = _sigma_sieve(limit)
            # Build map: v -> list of k
            groups = defaultdict(list)
            max_M = 10**18
            for k in range(1, limit + 1):
                v = k * sigma[k]
                if v > max_M:
                    continue
                # Check k <= sqrt(M): k^2 <= v = k*sigma(k) => k <= sigma(k)
                if k > sigma[k]:
                    continue
                groups[v].append(k)

            # Find best group
            for v, ks in groups.items():
                if len(ks) > best_count:
                    # Verify all k_i <= isqrt(v)
                    sq = math.isqrt(v)
                    valid_ks = [k for k in ks if k <= sq]
                    if len(valid_ks) > best_count:
                        best_count = len(valid_ks)
                        best_M = v
                        best_k = valid_ks[:1000]  # at most 1000
            del groups
            del sigma
        except (MemoryError, Exception):
            break

    # Verify with exact factorization for safety
    verified = []
    for k in best_k:
        if k * _sigma_exact(k) == best_M and k <= math.isqrt(best_M):
            verified.append(k)
    
    if len(verified) > 0:
        best_k = verified
    else:
        # Fallback to known solution
        best_M = 333312
        best_k = [336, 372, 434]

    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k[:1000]]

    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
