import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # Deterministic Miller-Rabin for n < 2^64
    _MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in _MR_BASES:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Sieve primes up to 10^6 for trial division
    def sieve_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_p in enumerate(sieve) if is_p]

    small_primes = sieve_primes(1000000)

    def is_p_smooth(n, p, primes):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        m = n
        # Divide by small primes
        for q in primes:
            if q > p:
                break
            if q * q > m:
                break
            while m % q == 0:
                m //= q
        if m == 1:
            return True
        if is_prime(m):
            return m <= p
        return False

    best_k = 0
    best_p = None

    # Phase 1: Check small primes
    # We can generate primes up to 10^6 using the sieve list
    for p in small_primes:
        if time.time() - start_time > DEADLINE:
            break
        if p > 10**6: # small_primes only goes to 10^6
            break
        k = 0
        base = p * p
        while k < 200:
            if time.time() - start_time > DEADLINE:
                break
            n = base + k + 1
            if not is_p_smooth(n, p, small_primes):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
            if k >= 200:
                break

    # Phase 2: Check larger primes
    # Generate primes in ranges using Miller-Rabin
    # We'll scan odd numbers and test primality
    # To avoid missing primes, we can use a segmented sieve or just test candidates
    
    # Strategy: Scan ranges of odd numbers
    # For p in [10^6, 10^7], [10^7, 10^8], etc.
    
    # To speed up, we can skip multiples of 3, 5, etc.
    # But Miller-Rabin is fast enough for a few thousand candidates.
    
    ranges = [
        (1000000, 2000000),
        (2000000, 5000000),
        (5000000, 10000000),
        (10000000, 50000000),
        (50000000, 100000000)
    ]
    
    for lo, hi in ranges:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Sample candidates
        # Step size to limit number of candidates
        # If range is 10^6, step 100 gives 10^4 candidates
        # Miller-Rabin is ~1us? 10^4 us = 10ms.
        # Smoothness check is slower.
        # Let's try step 10 for small ranges, larger for big ones.
        
        step = max(2, (hi - lo) // 5000) # ~5000 candidates per range
        # Ensure step is even to hit odds if lo is even
        if step % 2 == 0:
            start = lo + 1 if lo % 2 == 0 else lo
        else:
            start = lo + 1 if lo % 2 == 0 else lo
            
        for p in range(start, hi, step):
            if time.time() - start_time > DEADLINE - 2:
                break
            if p > 10**9:
                break
            if not is_prime(p):
                continue
            
            k = 0
            base = p * p
            while k < 200:
                if time.time() - start_time > DEADLINE - 1:
                    break
                n = base + k + 1
                if not is_p_smooth(n, p, small_primes):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
                if k >= 200:
                    break
        if best_k >= 200:
            break

    if best_p is None:
        best_p = 2
        best_k = 1

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
