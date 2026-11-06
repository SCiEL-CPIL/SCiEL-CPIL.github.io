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

def phi_segment(start, length, primes):
    """Compute phi(n) for n in [start, start+length-1] using segmented sieve."""
    phi_vals = list(range(start, start + length))
    remaining = list(range(start, start + length))
    
    max_val = start + length - 1
    sqrt_max = int(math.isqrt(max_val))
    
    for p in primes:
        if p > sqrt_max:
            break
        first = ((start + p - 1) // p) * p
        for m in range(first, start + length, p):
            idx = m - start
            phi_vals[idx] -= phi_vals[idx] // p
            while remaining[idx] % p == 0:
                remaining[idx] //= p
    
    for i in range(length):
        if remaining[i] > 1:
            phi_vals[i] -= phi_vals[i] // remaining[i]
    
    return phi_vals

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Leave 2s safety margin
    
    # Precompute primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    def is_valid_run(s, k):
        phi_vals = phi_segment(s, k, primes)
        return len(set(phi_vals)) == k

    # Try multiple candidate positions for k=200
    candidates = []
    
    # Near 10^12
    for offset in range(0, 3000):
        candidates.append(10**12 - 199 - offset)
    
    # Near small values
    for s in range(1, 10000):
        candidates.append(s)
    
    # Random positions across the range
    random.seed(42)
    for _ in range(10000):
        s = random.randint(1, 10**12 - 199)
        candidates.append(s)
    
    # Remove duplicates
    candidates = list(dict.fromkeys(candidates))
    
    # Try each candidate for k=200
    for s in candidates:
        if time.time() > deadline - 5:
            break
        if s < 1 or s + 199 > 10**12:
            continue
        if is_valid_run(s, 200):
            return {'start': s, 'length': 200}
    
    # If k=200 not found, try smaller k values
    for k in range(199, 1, -1):
        if time.time() > deadline - 2:
            break
        for s in candidates:
            if s < 1 or s + k - 1 > 10**12:
                continue
            if is_valid_run(s, k):
                return {'start': s, 'length': k}
    
    # Fallback: return a known valid small run
    return {'start': 4, 'length': 2}
