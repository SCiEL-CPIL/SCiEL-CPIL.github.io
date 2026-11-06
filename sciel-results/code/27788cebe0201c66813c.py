"""Improved program using precomputed primes and optimized p-smooth check."""
import math
import time
import bisect

def sieve_primes(limit):
    """Generate list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    time_limit = 115
    
    # Precompute primes up to 10^7
    SIEVE_LIMIT = 10**7
    primes = sieve_primes(SIEVE_LIMIT)
    
    best_k = 0
    best_p = 2
    
    # Iterate through primes
    for i, p in enumerate(primes):
        if time.time() - start_time > time_limit:
            break
        
        k = 0
        for j in range(1, 201):
            v = p * p + j
            rem = v
            if rem <= p:
                k = j
                continue
            
            # Check if rem is p-smooth by dividing by primes <= p
            # Since primes is sorted and p is primes[i], we only need primes[0..i]
            smooth = True
            for q in primes:
                if q > p:
                    break
                if q > rem:
                    break
                if q * q > rem:
                    break
                while rem % q == 0:
                    rem //= q
                    if rem == 1:
                        break
            if rem <= p:
                k = j
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
        if k == 200:
            break
            
    return {'p': best_p, 'k': best_k}
