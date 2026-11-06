"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

DEADLINE = 85.0
_start = time.time()

def _elapsed():
    return time.time() - _start

def _sieve_spf(max_n):
    """Build smallest prime factor array up to max_n using vectorized sieve."""
    spf = np.arange(max_n + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(max_n)) + 1):
        if spf[i] == i:
            # Vectorized: set spf for multiples of i starting from i*i
            start = i * i
            spf[start::i] = np.minimum(spf[start::i], i)
    return spf

def _precompute_powerful_parts_vectorized(max_n):
    """
    Precompute powerful part for all numbers 1..max_n using a sieve-like approach.
    For each prime power p^e (e>=2), multiply p^e into all multiples of p^e
    that are NOT multiples of p^(e+1).
    """
    pp = np.ones(max_n + 1, dtype=np.int64)
    
    # Sieve to find primes
    is_prime = np.ones(max_n + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    for p in primes:
        p = int(p)
        pe = p * p
        if pe > max_n:
            break
        while pe <= max_n:
            # Multiply pe into all multiples of pe that are NOT multiples of pe*p
            # Multiples of pe: pe, 2*pe, 3*pe, ..., k*pe where k*pe <= max_n
            # Exclude those where k is a multiple of p (i.e., number is divisible by pe*p)
            # So k in {1, 2, ..., max_n//pe} where k % p != 0
            
            max_k = max_n // pe
            # Create indices: k*pe for k in 1..max_k where k % p != 0
            # Use boolean mask
            k_vals = np.arange(1, max_k + 1)
            mask = (k_vals % p) != 0
            indices = k_vals[mask] * pe
            
            pp[indices] *= pe
            
            if _elapsed() > DEADLINE - 5:
                break
            
            pe_next = pe * p
            if pe_next > max_n:
                break
            pe = pe_next
    
    return pp

def _precompute_powerful_parts_spf(max_n, spf):
    """Precompute powerful part for all numbers 1..max_n using SPF factorization."""
    pp = np.ones(max_n + 1, dtype=np.int64)
    for n in range(2, max_n + 1):
        if _elapsed() > DEADLINE:
            break
        temp = n
        ppart = 1
        while temp > 1:
            p = spf[temp]
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if e >= 2:
                ppart *= p ** e
        pp[n] = ppart
    return pp

def construct():
    global _start, DEADLINE
    _start = time.time()
    DEADLINE = 85.0
    
    max_n = 1000000
    min_n = 10000
    
    # Use the vectorized sieve approach - it's much faster
    # For each prime power p^e (e>=2), multiply into appropriate multiples
    pp = _precompute_powerful_parts_vectorized(max_n + 1)
    
    if _elapsed() > DEADLINE:
        return min_n
    
    # Direct scan of all consecutive pairs - finds the GLOBAL OPTIMUM
    # pp[n] * pp[n+1] for n in [min_n, max_n]
    scores = pp[min_n:max_n+1] * pp[min_n+1:max_n+2]
    idx = int(np.argmax(scores))
    best_score = int(scores[idx])
    best_n = min_n + idx
    
    return best_n

# EVOLVE-BLOCK-END
