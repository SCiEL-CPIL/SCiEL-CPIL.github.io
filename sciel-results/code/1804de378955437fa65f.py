import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    # 1. Generate primes up to 10^6 using a fast bytearray sieve
    limit_p = 10**6
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i, val in enumerate(is_prime) if val]
    # Small primes for quick divisibility checks
    small_primes = [p for p in primes if p < 1000]

    # 2. Efficient phi calculation using precomputed primes
    def phi(x):
        if x == 1:
            return 1
        result = x
        n = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        if n > 1:
            result -= result // n
        return result

    def phi_fast(n):
        """Faster phi for large n using small primes first, then trial division."""
        if n == 1:
            return 1
        result = n
        m = n
        # Try small primes first
        for p in small_primes:
            if p * p > m:
                break
            if m % p == 0:
                while m % p == 0:
                    m //= p
                result -= result // p
        # Then larger primes
        for p in primes:
            if p < 1000:
                continue
            if p * p > m:
                break
            if m % p == 0:
                while m % p == 0:
                    m //= p
                result -= result // p
        if m > 1:
            result -= result // m
        return result

    def check_run(s, target_len=200):
        seen = set()
        k = 0
        for i in range(target_len):
            val = phi_fast(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k

    best_k = 1
    best_s = 1

    # Strategy: Focus on large numbers where phi collisions are less likely.
    # The key insight: for random large n, phi(n) is "random-like" and collisions
    # in a window of 200 are unlikely. We need to find a window where all 200 
    # values are distinct.
    
    # Phase 1: Dense sweep near 10^12
    # Near 10^12, numbers are large, so phi(n) values spread out more.
    # Check consecutive blocks.
    base_start = 10**12 - 200
    for offset in range(0, 50000, 10):
        s = base_start - offset
        if s < 1:
            break
        k = check_run(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                return {'start': best_s, 'length': best_k}
        if time.time() - start_time > DEADLINE:
            break

    # Phase 2: Check specific structured locations
    # Numbers that are products of large primes, or near prime powers
    structured_starts = [
        10**12 - 199,
        10**12 - 200,
        10**12 - 1000,
        10**12 - 5000,
        10**12 - 10000,
        5 * 10**11,
        7 * 10**11,
        3 * 10**11,
        10**11,
        999999999989,  # Near 10^12
        999999999999,
        999999999990,
        999999999900,
        999999999000,
        999999990000,
        999999900000,
        999999000000,
        999990000000,
        999900000000,
        999000000000,
    ]
    
    for s in structured_starts:
        if s < 1 or s + 199 > 10**12:
            continue
        k = check_run(s, 200)
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                return {'start': best_s, 'length': best_k}
        if time.time() - start_time > DEADLINE:
            break

    # Phase 3: Randomized search with adaptive sampling
    # Focus on upper range where collisions are rarer
    random.seed(12345)
    attempts = 0
    max_attempts = 200000
    
    # Large primes for structured sampling
    large_primes = [p for p in primes if p > 100000]
    
    while time.time() - start_time < DEADLINE and attempts < max_attempts:
        # Weighted random sampling favoring large numbers
        r = random.random()
        if r < 0.4:
            # Very close to 10^12
            s = random.randint(10**12 - 10**7, 10**12 - 199)
        elif r < 0.7:
            # Upper range
            s = random.randint(10**11, 10**12 - 199)
        elif r < 0.85:
            # Mid-upper range
            s = random.randint(10**10, 10**11 - 199)
        elif r < 0.95:
            # Structured: near large prime multiples
            if large_primes:
                base = random.choice(large_primes)
                s = random.randint(1, 10**12 // base) * base + random.randint(0, base - 1)
                if s < 1:
                    s = 1
                if s + 199 > 10**12:
                    s = 10**12 - 199
            else:
                s = random.randint(10**11, 10**12 - 199)
        else:
            # Lower range (less promising but for completeness)
            s = random.randint(10**8, 10**10 - 199)
        
        k = check_run(s, 200)
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                return {'start': best_s, 'length': best_k}
        
        attempts += 1
        
        # Check time every 10 attempts
        if attempts % 10 == 0 and time.time() - start_time > DEADLINE:
            break

    return {'start': best_s, 'length': best_k}
