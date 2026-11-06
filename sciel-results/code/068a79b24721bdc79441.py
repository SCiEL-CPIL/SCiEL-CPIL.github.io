import time
import math

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
    deadline = start_time + 118
    
    # Precompute primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    
    def is_valid_run(s, k):
        seen = set()
        for i in range(k):
            n = s + i
            v = phi_small(n, primes)
            if v in seen:
                return False
            seen.add(v)
        return True

    # Try to find a run of length 200
    # Start from the largest possible s
    s_candidate = 10**12 - 199
    if is_valid_run(s_candidate, 200):
        return {'start': s_candidate, 'length': 200}
        
    # If it failed, try a few others
    for offset in range(1, 1000):
        if time.time() > deadline:
            break
        s = 10**12 - 199 - offset
        if s < 1:
            break
        if is_valid_run(s, 200):
            return {'start': s, 'length': 200}
            
    return {'start': 1, 'length': 1}
