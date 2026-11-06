"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_SMALL_PRIMES_SET = set(_SMALL_PRIMES)

def _extract_small_factor(x):
    """Return the part of x composed only of primes < 24, and the set of those primes used."""
    factor = x
    primes_used = set()
    for p in _SMALL_PRIMES:
        if factor % p == 0:
            primes_used.add(p)
            while factor % p == 0:
                factor //= p
    return factor, primes_used

def _score(n):
    """Count distinct small primes across n+1..n+24."""
    all_primes = set()
    for i in range(1, 25):
        _, ps = _extract_small_factor(n + i)
        all_primes |= ps
    return len(all_primes)

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    best_n = 1000
    best_score = _score(1000)
    
    # First pass: scan all n in range, keep best
    n = 1000
    while n <= 1000000:
        if time.time() > deadline:
            break
        s = _score(n)
        if s < best_score:
            best_score = s
            best_n = n
            if best_score <= 2:
                # Very good, do a quick local check then return
                # Check neighbors for even better
                for dn in range(-50, 51):
                    nn = n + dn
                    if 1000 <= nn <= 1000000 and nn != n:
                        ss = _score(nn)
                        if ss < best_score:
                            best_score = ss
                            best_n = nn
                if best_score == 1:
                    return best_n
                break
        n += 1
    
    # If we found score <= 3, do a more targeted search
    if best_score <= 3:
        # Try to find score 1 or 2 with more targeted approach
        # Look for n where all n+i have small-prime part as power of 2 (or few primes)
        # This means n+i = 2^a * (large prime factor) for various a
        # Or n+i = 3^a * (large prime factor), etc.
        
        # Second pass: check all n again but with early termination
        for n in range(1000, 1000001):
            if time.time() > deadline:
                break
            if n == best_n:
                continue
            s = _score(n)
            if s < best_score:
                best_score = s
                best_n = n
                if best_score == 1:
                    return best_n
    
    return best_n
# EVOLVE-BLOCK-END
