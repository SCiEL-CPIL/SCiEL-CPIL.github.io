import time
import math
import sys
import random

# Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    # Small primes check
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Miller-Rabin
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        if a % n == 0:
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

def sieve(n):
    """Return a list of primes up to n."""
    if n < 2:
        return []
    is_prime_arr = [True] * (n + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i*i, n+1, i):
                is_prime_arr[j] = False
    return [i for i, prime in enumerate(is_prime_arr) if prime]

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    
    y, c, m = random.randint(1, n-1), random.randint(1, n-1), random.randint(1, n-1)
    g, r, q = 1, 1, 1
    while g == 1:
        x = y
        for i in range(r):
            y = (y*y + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            for i in range(min(m, r-k)):
                y = (y*y + c) % n
                q = (q * abs(x-y)) % n
            g = math.gcd(q, n)
            k += m
        r *= 2
    if g == n:
        while True:
            ys = (ys*ys + c) % n
            g = math.gcd(abs(x-ys), n)
            if g != 1:
                break
    return g

def factor(n, factors):
    if n == 1:
        return
    if is_prime(n):
        factors.append(n)
        return
    d = pollard_brent(n)
    factor(d, factors)
    factor(n//d, factors)

def largest_prime_factor(n):
    if n == 1:
        return 1
    factors = []
    factor(n, factors)
    return max(factors)

def _largest_pf(n):
    if n == 1:
        return 1
    factors = []
    factor(n, factors)
    return max(factors)

def has_all_factors_at_most_p(n, p):
    if n <= 1:
        return True
    # If n is prime, it's only p-smooth if n <= p
    if is_prime(n):
        return n <= p
    
    # We want to check if all prime factors of n are <= p.
    # This is equivalent to: largest_prime_factor(n) <= p.
    # But factorization is expensive.
    # Optimization: 
    # 1. Trial divide by small primes.
    # 2. If remainder is 1, True.
    # 3. If remainder is prime, check <= p.
    # 4. Else, check if remainder has a factor > p.
    
    rem = n
    # Use global small_primes
    global _small_primes
    for q in _small_primes:
        if q > p:
            break
        if q * q > rem:
            break
        while rem % q == 0:
            rem //= q
            
    if rem == 1:
        return True
        
    if is_prime(rem):
        return rem <= p
        
    # rem is composite.
    # It has no prime factors in _small_primes (up to min(p, sqrt(rem))).
    # If we broke because q > p, then any remaining factor > p?
    # If we broke because q*q > rem, then rem is prime (handled above).
    # So we must have broken because q > p OR we exhausted _small_primes.
    
    # If p is large, we might have exhausted _small_primes (up to 1e5).
    # rem has factors > 1e5.
    # We need to check if rem has a factor > p.
    # If p >= 1e5, and rem is composite, it has at least two factors > 1e5.
    # Let d be the smallest prime factor of rem.
    # If d > p, return False.
    # If d <= p, then rem/d might have a factor > p.
    
    # Use Pollard-Brent to find a factor of rem.
    d = pollard_brent(rem)
    # Check d and rem//d recursively?
    # Or just check if largest prime factor of rem <= p.
    # Since rem is not too large (n ~ p^2, p ~ 1e9, n ~ 1e18),
    # factorization should be fast.
    
    # To avoid deep recursion or overhead, we can just find the largest prime factor.
    # But we already have a function for that.
    # However, we can be smarter.
    # If d > p, return False.
    # If d <= p, check rem//d.
    
    if d > p:
        return False
    return has_all_factors_at_most_p(rem // d, p)

_SMALL_PRIMES_LIMIT = 100000
_small_primes = None

def construct():
    global _small_primes
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if _small_primes is None:
        _small_primes = sieve(_SMALL_PRIMES_LIMIT)
        
    best_k = 0
    best_p = 3
    
    # Random sampling of primes
    # We sample primes in [10^6, 10^9]
    # Also check some small primes just in case
    
    # First, check small primes quickly
    # Sieve up to 10000
    small_check_primes = sieve(10000)
    for p in small_check_primes:
        if time.time() > deadline - 5:
            break
        base = p * p
        k = 0
        for i in range(1, 201):
            if time.time() > deadline - 5:
                break
            n = base + i
            if has_all_factors_at_most_p(n, p):
                k = i
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # Now random sampling
    # To generate a random prime, we pick a random number and check
    # We can bias towards larger numbers if we want, but uniform is fine.
    
    # Number of iterations depends on speed.
    # Let's try 5000 iterations or until deadline.
    
    while time.time() < deadline - 5:
        # Pick a random odd number in range
        # Range: 10^6 to 10^9
        # random.randint(10**6, 10**9)
        # Ensure it's odd
        p_candidate = random.randint(10**6, 10**9)
        if p_candidate % 2 == 0:
            p_candidate += 1
        
        # Check if prime
        # We can do a few trial divisions first to speed up
        is_p_prime = True
        for q in _small_primes:
            if q * q > p_candidate:
                break
            if p_candidate % q == 0:
                is_p_prime = False
                break
        
        if not is_p_prime:
            if p_candidate > 2:
                # Check Miller Rabin if trial division passed
                # But trial division by 1e5 primes is fast enough?
                # 1e5 primes is ~100k ops. 
                # Miller Rabin is ~log(n) ops.
                # Let's just use is_prime
                if is_prime(p_candidate):
                    is_p_prime = True
            else:
                is_p_prime = False
        
        if not is_p_prime:
            continue
            
        p = p_candidate
        base = p * p
        k = 0
        # Check up to 200
        for i in range(1, 201):
            if time.time() > deadline - 5:
                break
            n = base + i
            if has_all_factors_at_most_p(n, p):
                k = i
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
        # If we found a good k, we might stop early?
        # No, keep searching to maximize.
        
    return {'p': best_p, 'k': best_k}
