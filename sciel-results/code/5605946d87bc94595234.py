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

def _precompute_powerful_parts_fast(N, spf):
    """
    Efficient powerful-part computation using a sieve approach.
    For each prime p, for each power p^e (e>=2), multiply p^e into
    all multiples of p^e that are NOT multiples of p^(e+1).
    """
    pp = np.ones(N + 1, dtype=np.int64)
    
    # Find all primes up to N
    primes = []
    for i in range(2, N + 1):
        if spf[i] == i:  # i is prime
            primes.append(i)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        pe = p2
        while pe <= N:
            # Indices to update: multiples of pe, but not multiples of pe*p
            # k*pe where k is not divisible by p
            # We can iterate k from 1 to N//pe
            # Optimization: The step is pe. The indices are pe, 2pe, 3pe...
            # We want to exclude indices that are multiples of pe*p.
            # This is equivalent to:
            # for k in range(1, N // pe + 1):
            #     if k % p != 0:
            #         pp[k * pe] *= pe
            
            # Vectorized approach:
            # Create an array of indices 1..N//pe
            # Mask out multiples of p
            # This might be overkill for small N/pe, but let's try a simple loop first
            # Given the small total count (~450k operations), a simple loop is sufficient and likely faster than creating numpy arrays for each power.
            
            # Let's stick to the loop for simplicity and reliability, it's very fast.
            limit = N // pe
            for k in range(1, limit + 1):
                if k % p != 0:
                    pp[k * pe] *= pe
            
            pe *= p
    
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
    # Note: pp values can be large, but fit in int64?
    # Max powerful part for N=1e6:
    # 2^19 = 524288. 3^12 = 531441.
    # Product of two such could be ~ 10^12, which fits in int64 (9e18).
    products = pp[LO:N] * pp[LO+1:N+1]
    best_idx = np.argmax(products)
    best_n = LO + best_idx
    best_val = products[best_idx]
    
    elapsed = time.time() - t0
    if elapsed > _DEADLINE - 2:
        return int(best_n)
    
    return int(best_n)
# EVOLVE-BLOCK-END
