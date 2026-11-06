import time
import math

def sieve_primes(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def phi_fast(n, primes):
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
    deadline = start_time + 140
    
    primes = sieve_primes(10**6)
    
    # Try to find a run of length 200
    # Start near 10^12
    s = 10**12 - 200
    
    while s >= 1 and time.time() < deadline:
        seen = set()
        valid = True
        for i in range(200):
            n = s + i
            v = phi_fast(n, primes)
            if v in seen:
                valid = False
                break
            seen.add(v)
        
        if valid:
            return {'start': s, 'length': 200}
        
        s -= 1
        
    # Fallback: if we couldn't find 200, try smaller k?
    # Unlikely to happen.
    # Just return a valid small run.
    s = 10**12 - 200
    seen = set()
    k = 0
    for i in range(200):
        n = s + i
        v = phi_fast(n, primes)
        if v in seen:
            break
        seen.add(v)
        k += 1
    if k > 0:
        return {'start': s, 'length': k}
    return {'start': 1, 'length': 1}
