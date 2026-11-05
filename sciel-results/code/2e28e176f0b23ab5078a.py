import math
import time

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, val in enumerate(is_prime) if val]

# Precompute primes up to 10^6. sqrt(10^12) = 10^6.
# This is fast and done once.
PRIMES = _get_primes(10**6)

def construct():
    start_time = time.time()
    LIMIT = 110.0  # Safety margin for 120s limit
    MAX_S = 10**12
    MAX_K = 200
    
    primes = PRIMES
    primes_len = len(primes)
    
    def phi(x):
        if x == 1:
            return 1
        result = x
        m = x
        # Check 2 separately for slight speedup
        if m % 2 == 0:
            result -= result // 2
            while m % 2 == 0:
                m //= 2
        # Check odd primes
        for i in range(1, primes_len):
            p = primes[i]
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    # Strategy:
    # Collisions in phi(n) for n ~ 10^12 are extremely rare in a window of 200.
    # We simply pick a start near the upper bound and verify.
    # If the window is valid, we return k=200 immediately.
    
    # Try a deterministic sequence of windows near 10^12
    # Start from a bit below 10^12 to be safe
    base_start = MAX_S - 1000
    
    for offset in range(0, 1000):
        if time.time() - start_time > LIMIT:
            break
            
        s = base_start + offset
        if s + MAX_K - 1 > MAX_S:
            continue
            
        # Check if phi(s) ... phi(s+199) are distinct
        seen = set()
        valid = True
        for i in range(s, s + MAX_K):
            val = phi(i)
            if val in seen:
                valid = False
                break
            seen.add(val)
            
        if valid:
            return {'start': s, 'length': MAX_K}

    # Fallback: Random search if deterministic fails (should not happen)
    import random
    max_start = MAX_S - MAX_K + 1
    while time.time() - start_time < LIMIT:
        s = random.randint(1, max_start)
        seen = set()
        valid = True
        for i in range(s, s + MAX_K):
            val = phi(i)
            if val in seen:
                valid = False
                break
            seen.add(val)
        if valid:
            return {'start': s, 'length': MAX_K}

    # Last resort: return a valid small k (e.g., k=1 is always valid)
    return {'start': 1, 'length': 1}
