import math
import time
import random

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

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    
    x = 2
    y = 2
    d = 1
    c = 1
    # Use a fixed number of iterations for c=1, then increase c
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = math.gcd(abs(x - y), n)
        if d == n:
            c += 1
            x = 2
            y = 2
            d = 1
            if c > 10:
                # Fallback to random start if stuck
                x = random.randint(2, n-2)
                y = x
                c = 1
    return d

# Precompute small primes for trial division
SMALL_PRIMES = []
def _gen_small_primes(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

SMALL_PRIMES = _gen_small_primes(10000)

def is_p_smooth(n, p):
    """
    Returns True if all prime factors of n are <= p.
    """
    if n <= 1:
        return True
    if n <= p:
        return True
    
    # Trial division by small primes
    # We only divide by primes q <= p
    for q in SMALL_PRIMES:
        if q > p:
            break
        if q * q > n:
            break
        while n % q == 0:
            n //= q
        if n == 1:
            return True
        if n <= p:
            return True
            
    # If n is prime, check if it's <= p
    if is_prime(n):
        return n <= p
        
    # Use Pollard-Rho to find a factor
    d = pollard_rho(n)
    if d == n:
        # Should be handled by is_prime above, but just in case
        return n <= p
        
    # Recursively check factors
    return is_p_smooth(d, p) and is_p_smooth(n // d, p)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Sieve primes up to 200,000 for search space
    # This keeps p^2 around 4*10^10, which is manageable for Pollard-Rho
    limit = 200000
    is_prime_sieve = bytearray(b'\x01') * (limit + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_sieve[i]:
            is_prime_sieve[i*i::i] = bytearray(len(is_prime_sieve[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime_sieve[i]]
    
    # Search strategy:
    # Iterate through primes. 
    # To maximize k, we want to find a p where the sequence p^2+1, ..., p^2+k is smooth.
    # We check k up to 200.
    # We can speed this up by checking the first few terms quickly.
    
    # We'll iterate primes in increasing order? Or random?
    # Let's try increasing order, as smaller p might have "luckier" configurations?
    # Actually, density is similar. Let's just scan.
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Quick check for p^2+1
        # If p^2+1 is not p-smooth, k=0.
        # We can just call is_p_smooth for i=1.
        
        k = 0
        p_sq = p * p
        
        for i in range(1, 201):
            if time.time() > deadline:
                break
                
            n_val = p_sq + i
            
            if is_p_smooth(n_val, p):
                k = i
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
        # Optimization: If we've found a k close to 200, we might want to stop early?
        # No, we must maximize k. But if we find 200, we can stop.
        if best_k == 200:
            break

    return {'p': best_p, 'k': best_k}
