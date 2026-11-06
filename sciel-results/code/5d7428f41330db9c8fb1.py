"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
import numpy as np

_DEADLINE = 75.0  # seconds, safety margin before 90s kill

def _sieve_primes(N):
    """Sieve of Eratosthenes to get all primes up to N."""
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, N + 1) if is_prime[i]]

def _precompute_powerful_parts_sieve(N):
    """
    Sieve-based computation of powerful parts.
    For each prime p and each exponent e >= 2, for each multiple m of p^e,
    if the exponent of p in m is exactly e, multiply pp[m] by p^e.
    """
    pp = np.ones(N + 1, dtype=np.int64)
    primes = _sieve_primes(N)
    
    for p in primes:
        pe = p * p  # p^2
        e = 2
        while pe <= N:
            next_pe = pe * p  # p^(e+1)
            # For each multiple of pe, check if exponent is exactly e
            for m in range(pe, N + 1, pe):
                if m % next_pe != 0:
                    pp[m] *= pe
            pe = next_pe
            e += 1
    
    return pp

def _verify_powerful_part_brute(x):
    """Brute-force verification of powerful part for a single number."""
    if x < 4:
        return 1
    pp = 1
    y = x
    d = 2
    while d * d <= y:
        if y % d == 0:
            e = 0
            while y % d == 0:
                y //= d
                e += 1
            if e >= 2:
                pp *= d ** e
        d += 1
    if y > 1:
        # y is prime with exponent 1, doesn't contribute
        pass
    return pp

def construct():
    t0 = time.time()
    N = 1000000
    LO = 10000
    
    # Precompute powerful parts using sieve
    pp = _precompute_powerful_parts_sieve(N)
    
    elapsed = time.time() - t0
    if elapsed > _DEADLINE - 5:
        return LO
    
    # Find max of pp[n] * pp[n+1] for n in [LO, N-1]
    products = pp[LO:N] * pp[LO+1:N+1]
    best_idx = np.argmax(products)
    best_n = LO + best_idx
    best_val = products[best_idx]
    
    elapsed = time.time() - t0
    if elapsed > _DEADLINE - 3:
        return int(best_n)
    
    # Verify: double-check the powerful part computation for best_n and best_n+1
    v1 = _verify_powerful_part_brute(best_n)
    v2 = _verify_powerful_part_brute(best_n + 1)
    
    if v1 != pp[best_n] or v2 != pp[best_n + 1]:
        # Mismatch - fall back to SPF-based computation
        spf = np.zeros(N + 1, dtype=np.int64)
        for i in range(2, N + 1):
            if spf[i] == 0:
                spf[i::i] = i
        
        pp2 = np.ones(N + 1, dtype=np.int64)
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
            pp2[x] = val
        
        products2 = pp2[LO:N] * pp2[LO+1:N+1]
        best_idx2 = np.argmax(products2)
        best_n = LO + best_idx2
    
    return int(best_n)
# EVOLVE-BLOCK-END
