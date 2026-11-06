import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin

    # Sieve primes up to 10^7
    SIEVE_LIMIT = 10_000_000
    is_prime_sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = 0
    for i in range(2, int(math.isqrt(SIEVE_LIMIT)) + 1):
        if is_prime_sieve[i]:
            step = i
            start = i * i
            is_prime_sieve[start:SIEVE_LIMIT + 1:step] = bytearray(len(is_prime_sieve[start:SIEVE_LIMIT + 1:step]))
    
    primes = [i for i in range(2, SIEVE_LIMIT + 1) if is_prime_sieve[i]]
    n_primes = len(primes)
    
    best_k = 1
    best_p = 2

    def check_smoothness(p, L, R, primes_list, max_k=200):
        """Check how many consecutive numbers from L to R are p-smooth.
        Returns the count of consecutive p-smooth numbers starting at L."""
        if time.time() - start_time > DEADLINE:
            return -1
        
        window_size = min(max_k, R - L + 1)
        rem = [L + i for i in range(window_size)]
        
        # Sieve with primes q <= p
        for q in primes_list:
            if q > p:
                break
            if q > R:
                break
            
            # Find first multiple of q in [L, R]
            first = L + (q - L % q) % q
            
            for m in range(first, R + 1, q):
                idx = m - L
                if rem[idx] == 1:
                    continue
                while rem[idx] % q == 0:
                    rem[idx] //= q
                    
        # Count consecutive p-smooth numbers
        k = 0
        for i in range(window_size):
            if rem[i] == 1:
                k += 1
            else:
                break
        return k

    # Sequential scan of small primes first (they're cheap to verify)
    for idx in range(min(5000, n_primes)):
        if time.time() - start_time > DEADLINE:
            break
        p = primes[idx]
        p2 = p * p
        k = check_smoothness(p, p2 + 1, p2 + 200, primes, 200)
        if k > 0 and k > best_k:
            best_k = k
            best_p = p

    # For larger primes, use a smarter strategy:
    # Only check primes where p^2+1 is likely to be p-smooth.
    # p^2+1 is p-smooth iff all prime factors of p^2+1 are <= p.
    # Since p^2+1 < (p+1)^2, any prime factor > p must be > sqrt(p^2+1) ~ p.
    # So p^2+1 has a prime factor > p iff it has a prime factor in (p, p^2+1].
    # Equivalently, p^2+1 is p-smooth iff it's not divisible by any prime in (p, sqrt(p^2+1)].
    # But sqrt(p^2+1) ~ p, so we just need to check if p^2+1 has a prime factor > p.
    # This is equivalent to checking if p^2+1 is prime or has a large prime factor.
    
    # Strategy: For each candidate p, first check if p^2+1 is p-smooth by
    # attempting to factor it using primes up to p. If it fails, skip.
    
    # Use stratified random sampling of primes
    random.seed(12345)
    
    # Divide primes into strata and sample from each
    strata_size = 1000
    n_strata = (n_primes + strata_size - 1) // strata_size
    
    for _ in range(100000):
        if time.time() - start_time > DEADLINE:
            break
        
        # Pick a random stratum and a random prime from it
        stratum = random.randint(0, n_strata - 1)
        lo = stratum * strata_size
        hi = min(lo + strata_size, n_primes)
        idx = random.randint(lo, hi - 1)
        
        p = primes[idx]
        p2 = p * p
        
        # Quick check: factor p^2+1 using primes up to p
        n = p2 + 1
        temp = n
        is_smooth = True
        for q in primes:
            if q > p:
                break
            if q * q > temp:
                break
            while temp % q == 0:
                temp //= q
            if temp == 1:
                break
        if temp != 1:
            continue  # p^2+1 is not p-smooth
        
        # Full check
        k = check_smoothness(p, p2 + 1, p2 + 200, primes, 200)
        if k > 0 and k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}
