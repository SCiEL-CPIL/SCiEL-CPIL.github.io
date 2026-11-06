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
    """Build smallest prime factor array up to max_n."""
    spf = np.arange(max_n + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(max_n)) + 1):
        if spf[i] == i:
            spf[i*i::i] = np.where(spf[i*i::i] == np.arange(i*i, max_n+1, i), i, spf[i*i::i])
    return spf

def _precompute_powerful_parts(max_n, spf):
    """Precompute powerful part for all numbers 1..max_n using SPF."""
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

def _generate_powerful_numbers(max_val):
    """Generate all powerful numbers (products of distinct prime powers p^e, e>=2) up to max_val."""
    limit = int(math.isqrt(max_val)) + 2
    if limit < 2:
        limit = 2
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    prime_powers = {}
    for p in primes:
        powers = []
        val = p * p
        e = 2
        while val <= max_val and e <= 12:
            powers.append(val)
            e += 1
            val *= p
        prime_powers[p] = powers
    
    active_primes = [p for p in primes if p * p <= max_val]
    
    current = [1]
    for p in active_primes:
        if _elapsed() > DEADLINE:
            break
        powers = prime_powers.get(p, [])
        if not powers:
            continue
        new_current = list(current)
        for pw in powers:
            for v in current:
                prod = v * pw
                if prod <= max_val:
                    new_current.append(prod)
        current = new_current
    
    powerful_list = sorted(set(current))
    return powerful_list

def _solve_crt_zero_minus1(A, B):
    """Solve x ≡ 0 (mod A), x ≡ -1 (mod B). Returns (x0, lcm) or None."""
    g = math.gcd(A, B)
    if (B - 1) % g != 0:
        return None
    a1 = A // g
    b1 = B // g
    c1 = (B - 1) // g
    inv_a1 = pow(a1, -1, b1)
    k0 = (c1 * inv_a1) % b1
    x0 = A * k0
    lcm = A * b1
    return x0, lcm

def construct():
    global _start, DEADLINE
    _start = time.time()
    DEADLINE = 85.0
    
    best_score = 0
    best_n = 10000
    
    max_n = 1000000
    min_n = 10000
    
    # Build SPF
    spf = _sieve_spf(max_n + 2)
    if _elapsed() > DEADLINE:
        return 10000
    
    # Precompute powerful parts for all numbers up to max_n+1
    pp = _precompute_powerful_parts(max_n + 1, spf)
    if _elapsed() > DEADLINE:
        return 10000
    
    # Direct scan of all consecutive pairs - this finds the GLOBAL OPTIMUM
    # pp[n] * pp[n+1] for n in [min_n, max_n]
    scores = pp[min_n:max_n+1] * pp[min_n+1:max_n+2]
    idx = int(np.argmax(scores))
    best_score = int(scores[idx])
    best_n = min_n + idx
    
    if _elapsed() > DEADLINE:
        return best_n
    
    # Generate powerful numbers for CRT-based refinement
    powerful_nums = _generate_powerful_numbers(max_n)
    if _elapsed() > DEADLINE:
        return best_n
    
    powerful_nums = [x for x in powerful_nums if x > 1]
    powerful_nums.sort(reverse=True)
    
    # CRT-based refinement: try to find n where powerful_part(n) >= A and powerful_part(n+1) >= B
    # with A*B > best_score, using top powerful numbers
    K = min(len(powerful_nums), 300)
    candidates = powerful_nums[:K]
    
    for i, A in enumerate(candidates):
        if _elapsed() > DEADLINE:
            break
        for j, B in enumerate(candidates):
            if _elapsed() > DEADLINE:
                break
            if i == j:
                continue
            if A * B <= best_score:
                continue
            if math.gcd(A, B) != 1:
                continue
            
            sol = _solve_crt_zero_minus1(A, B)
            if sol is None:
                continue
            x0, lcm = sol
            
            if x0 < min_n:
                t = (min_n - x0 + lcm - 1) // lcm
                n_val = x0 + t * lcm
            else:
                n_val = x0
            
            if n_val < min_n or n_val > max_n:
                continue
            
            actual_score = int(pp[n_val] * pp[n_val + 1])
            if actual_score > best_score:
                best_score = actual_score
                best_n = n_val
    
    return best_n

# EVOLVE-BLOCK-END
