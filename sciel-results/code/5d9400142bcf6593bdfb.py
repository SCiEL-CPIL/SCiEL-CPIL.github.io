"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
import numpy as np

_DEADLINE = 75.0  # seconds, safety margin before 90s kill

def _sieve_spf(n):
    """Smallest-prime-factor sieve up to n (inclusive)."""
    spf = np.zeros(n + 1, dtype=np.int64)
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i::i] = i
    return spf

def _powerful_part_from_spf(x, spf):
    """Compute powerful part of x using SPF table. Returns 1 if no prime power with e>=2."""
    if x < 4:
        return 1
    pp = 1
    while x > 1:
        p = spf[x]
        e = 0
        while x % p == 0:
            x //= p
            e += 1
        if e >= 2:
            pp *= p ** e
    return pp

def _precompute_powerful_parts_fast(N, spf):
    """
    For each number, factorize using SPF and compute powerful part.
    This is O(N log N) total which is fine for N=1e6.
    """
    pp = np.ones(N + 1, dtype=np.int64)
    for x in range(4, N + 1):
        y = x
        val = 1
        while y > 1:
            p = spf[y]
            e = 0
            while y % p == 0:
                y //= p
                e += 1
            if e >= 2:
                val *= p ** e
        pp[x] = val
    return pp

def construct():
    t0 = time.time()
    N = 1000000
    LO = 10000
    
    # Build SPF
    spf = _sieve_spf(N)
    
    elapsed = time.time() - t0
    if elapsed > _DEADLINE - 5:
        return LO
    
    # Precompute powerful parts
    pp = _precompute_powerful_parts_fast(N, spf)
    
    elapsed = time.time() - t0
    if elapsed > _DEADLINE - 3:
        return LO
    
    # Find max of pp[n] * pp[n+1] for n in [LO, N-1]
    products = pp[LO:N] * pp[LO+1:N+1]
    best_idx = np.argmax(products)
    best_n = LO + best_idx
    best_val = products[best_idx]
    
    elapsed = time.time() - t0
    if elapsed > _DEADLINE - 2:
        return int(best_n)
    
    # Verify: double-check the powerful part computation for best_n and best_n+1
    def _verify_powerful_part(x, spf):
        if x < 4:
            return 1
        pp = 1
        y = x
        while y > 1:
            p = spf[y]
            e = 0
            while y % p == 0:
                y //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        return pp
    
    v1 = _verify_powerful_part(best_n, spf)
    v2 = _verify_powerful_part(best_n + 1, spf)
    if v1 != pp[best_n] or v2 != pp[best_n + 1]:
        # Fallback: recompute all and find max
        pp2 = _precompute_powerful_parts_fast(N, spf)
        products2 = pp2[LO:N] * pp2[LO+1:N+1]
        best_idx2 = np.argmax(products2)
        best_n = LO + best_idx2
    
    return int(best_n)
# EVOLVE-BLOCK-END
