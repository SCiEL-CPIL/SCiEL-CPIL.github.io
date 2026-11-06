import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Deterministic Miller-Rabin for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
            return False
        for p in MR_BASES:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in MR_BASES:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def sieve_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_p in enumerate(sieve) if is_p]

    # Generate primes up to 10^7 for trial division
    PRIME_LIMIT = 10**7
    primes = sieve_primes(PRIME_LIMIT)
    
    # Helper to check if n is p-smooth
    def is_p_smooth(n, p):
        if n == 1:
            return True
        # If n is prime and n > p, it's not smooth
        # Since n = p^2 + i > p, if n is prime, it has a factor n > p.
        if is_prime(n):
            return False
        
        # Trial division by primes <= p
        # We only have primes up to PRIME_LIMIT.
        # If p > PRIME_LIMIT, we miss some primes.
        # But if p > PRIME_LIMIT, we can't fully check smoothness with this list.
        # However, if p is large, the number of primes <= p is large.
        # For now, assume p <= PRIME_LIMIT or handle carefully.
        # If p > PRIME_LIMIT, we can only check factors <= PRIME_LIMIT.
        # If remainder > 1, it might have factors in (PRIME_LIMIT, p].
        # But if p is large, this is a problem.
        # Let's restrict search to p <= PRIME_LIMIT for now, or handle the remainder.
        
        rem = n
        for q in primes:
            if q > p:
                break
            if q * q > rem:
                break
            if rem % q == 0:
                while rem % q == 0:
                    rem //= q
                if rem == 1:
                    return True
        # If we exited because q > p, then rem has no factors <= p.
        # So rem must be 1 (smooth) or a prime/product of primes > p.
        # If rem > 1, it has a factor > p.
        return rem == 1

    best_k = 0
    best_p = 2
    
    # Search primes
    # We iterate through the generated primes.
    for p in primes:
        if time.time() > deadline:
            break
            
        # Optional: skip very small p if we already have a good k?
        # No, small p might have long runs.
        
        k = 0
        base = p * p
        while k < 200:
            n = base + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p
            # If we found k=200, we can stop?
            # The max is 200.
            if k == 200:
                return {'p': best_p, 'k': best_k}

    return {'p': best_p, 'k': best_k}
