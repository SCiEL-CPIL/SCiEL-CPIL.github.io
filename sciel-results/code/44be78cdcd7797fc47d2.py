import math
import time

# Deterministic Miller-Rabin for n < 2^64
def is_prime(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _gen_primes(limit):
    """Generate primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

# We will search primes p up to 10^7.
# We need primes for trial division up to 10^7 as well.
# So we generate primes up to 10^7.
MAX_P = 10_000_000
PRIMES = _gen_primes(MAX_P)

def is_p_smooth(n, p, primes):
    """
    Returns True if all prime factors of n are <= p.
    """
    if n <= 1:
        return True
    if n <= p:
        return True
    
    # If n is prime, check if it's <= p
    if is_prime(n):
        return n <= p
        
    # Trial division by primes q <= p
    # We only need to divide by q while q*q <= n and q <= p
    for q in primes:
        if q > p:
            break
        if q * q > n:
            break
        while n % q == 0:
            n //= q
        if n == 1:
            return True
        if n <= p:
            # If n becomes <= p, it might be prime or composite.
            # But if it's composite, it has factors <= sqrt(n) <= sqrt(p) < p.
            # Wait, if n <= p, and we've divided by all primes <= p (up to sqrt(n)),
            # then n is prime.
            # Actually, if n <= p, and we haven't broken the loop because q*q > n,
            # then n is prime.
            # So we just return n <= p (which is True).
            return True
            
    # If we exit the loop:
    # 1. q > p: We have checked all primes <= p.
    #    If n > 1, then n has no prime factors <= p.
    #    Since n < p^2 (roughly), n must be prime.
    #    So n is a prime > p. Not smooth.
    # 2. q*q > n: We have checked all primes <= sqrt(n).
    #    So n is prime.
    #    We need to check if n <= p.
    #    But we broke because q*q > n.
    #    If n <= p, it's smooth.
    #    If n > p, it's not.
    #    Wait, if we break because q*q > n, we haven't necessarily checked q > p.
    #    But if q*q > n, then any factor of n must be >= q.
    #    So n is prime.
    #    So we just return n <= p.
    
    return n <= p

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Search primes p from the precomputed list
    # We iterate through primes p
    for p in PRIMES:
        if time.time() > deadline:
            break
            
        p_sq = p * p
        k = 0
        
        # Check consecutive integers p^2 + 1, ..., p^2 + 200
        for i in range(1, 201):
            if time.time() > deadline:
                break
                
            n_val = p_sq + i
            
            if is_p_smooth(n_val, p, PRIMES):
                k = i
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
        if best_k == 200:
            break

    return {'p': best_p, 'k': best_k}
