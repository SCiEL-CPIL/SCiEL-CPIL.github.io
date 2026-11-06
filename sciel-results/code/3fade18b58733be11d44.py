"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    MAX_N = 10**12
    
    def sieve_primes(limit):
        """Generate primes up to limit using sieve."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Precompute primes up to 10^6 (sqrt of 10^12)
    PRIME_LIMIT = 10**6
    primes = sieve_primes(PRIME_LIMIT)
    
    def phi(x):
        """Compute Euler's totient function phi(x) exactly using precomputed primes."""
        if x == 1:
            return 1
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    def check_run(start, max_length, max_n):
        """Check consecutive integers from start for distinct phi values.
        Returns length of longest run found."""
        seen = set()
        k = 0
        for i in range(max_length):
            n = start + i
            if n > max_n:
                break
            v = phi(n)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k
    
    best_k = 1
    best_s = 1
    
    # Phase 1: Dense search in small ranges
    search_limit = 200000
    for s in range(1, min(search_limit, int(MAX_N)) + 1):
        if time.time() > deadline - 10:
            break
        k = check_run(s, 200, MAX_N)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Phase 2: Search in larger ranges with sparse sampling
    if time.time() < deadline - 15:
        ranges = [
            (200000, 10**6),
            (10**6, 10**7),
            (10**7, 10**8),
            (10**8, 10**9),
            (10**9, 10**10),
            (10**10, 10**11),
            (10**11, 10**12),
        ]
        
        for lo, hi in ranges:
            if time.time() > deadline - 10:
                break
            num_samples = 500
            for _ in range(num_samples):
                if time.time() > deadline - 10:
                    break
                s = random.randint(lo, min(hi, int(MAX_N) - 200))
                k = check_run(s, 200, MAX_N)
                if k > best_k:
                    best_k = k
                    best_s = s
    
    # Phase 3: Local search around best found start
    if best_k > 5 and time.time() < deadline - 10:
        window = 5000
        for offset in range(-window, window + 1):
            if time.time() > deadline - 5:
                break
            s = best_s + offset
            if s < 1 or s + 199 > MAX_N:
                continue
            k = check_run(s, 200, MAX_N)
            if k > best_k:
                best_k = k
                best_s = s
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
