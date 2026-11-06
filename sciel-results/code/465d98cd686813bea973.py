import time
import math
import random

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i, v in enumerate(is_prime) if v]

def phi_small(n, primes):
    """Compute phi(n) using precomputed primes up to sqrt(n)."""
    result = n
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            result -= result // p
            while temp % p == 0:
                temp //= p
    if temp > 1:
        result -= result // temp
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Leave 2s safety margin
    
    # Precompute primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    def is_valid_run(s, k):
        seen = set()
        for i in range(k):
            n = s + i
            v = phi_small(n, primes)
            if v in seen:
                return False
            seen.add(v)
        return True

    # Try multiple candidate positions for k=200
    # Strategy: try positions near 10^12, near small values, and random positions
    candidates = []
    
    # Near 10^12
    for offset in range(0, 500):
        candidates.append(10**12 - 199 - offset)
    
    # Near small values
    for s in range(1, 1000):
        candidates.append(s)
    
    # Random positions across the range
    random.seed(42)
    for _ in range(200):
        s = random.randint(1, 10**12 - 199)
        candidates.append(s)
    
    # Remove duplicates and sort by priority (near 10^12 first, then small, then random)
    candidates = list(dict.fromkeys(candidates))
    
    # Try each candidate
    for s in candidates:
        if time.time() > deadline:
            break
        if s < 1 or s + 199 > 10**12:
            continue
        if is_valid_run(s, 200):
            return {'start': s, 'length': 200}
    
    # If k=200 not found, try smaller k values
    for k in range(199, 1, -1):
        if time.time() > deadline:
            break
        for s in candidates:
            if s < 1 or s + k - 1 > 10**12:
                continue
            if is_valid_run(s, k):
                return {'start': s, 'length': k}
    
    # Fallback: return a known valid small run
    return {'start': 4, 'length': 2}
