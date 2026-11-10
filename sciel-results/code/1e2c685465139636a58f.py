"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 145.0  # safety margin before 150s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Precompute primes up to 10^6 using sieve
    def sieve_primes(lim):
        is_prime = bytearray(b'\x01') * (lim + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(lim)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, lim + 1) if is_prime[i]]

    primes = sieve_primes(10**6)
    # Also keep a shorter list for quick trial division
    primes_small = [p for p in primes if p <= 10000]

    def phi_fast(n):
        if n == 1:
            return 1
        result = n
        m = n
        for p in primes_small:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    def find_run(s, max_k=200):
        seen = set()
        k = 0
        for i in range(max_k):
            v = phi_fast(s + i)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    best_k = 1
    best_s = 1

    # Phase 1: Dense range search [1, 500000]
    phase1_end = 500000
    vals = [0] * (phase1_end + 200)
    for x in range(1, phase1_end + 1):
        vals[x] = phi_fast(x)
        if x % 50000 == 0 and time_left() < 60:
            break

    # Find best run in Phase 1 range
    for s in range(1, phase1_end - 200):
        if time_left() < 60:
            break
        seen = set()
        k = 0
        while s + k <= phase1_end and vals[s + k] not in seen:
            seen.add(vals[s + k])
            k += 1
        if k > best_k:
            best_k = k
            best_s = s

    # Phase 2: Targeted local search around best_s and nearby
    # Try extending the best run found
    if time_left() > 20:
        for offset in range(-50, 51):
            if time_left() < 20:
                break
            s = best_s + offset
            if s < 1:
                continue
            k = find_run(s)
            if k > best_k:
                best_k = k
                best_s = s

    # Phase 3: Random sampling in [1, 10^12]
    random.seed(42)
    attempts = 0
    # First, try some structured ranges
    structured_ranges = [
        (10**6, 10**7),
        (10**7, 10**8),
        (10**8, 10**9),
        (10**9, 10**10),
        (10**10, 10**11),
        (10**11, 10**12),
    ]
    
    for lo, hi in structured_ranges:
        if time_left() < 10:
            break
        # Sample a few points in each range
        for _ in range(20):
            if time_left() < 10:
                break
            s = random.randint(lo, hi - 200)
            k = find_run(s)
            if k > best_k:
                best_k = k
                best_s = s
            attempts += 1

    # Phase 4: More random sampling
    while time_left() > 5:
        s = random.randint(1, 10**12 - 200)
        k = find_run(s)
        if k > best_k:
            best_k = k
            best_s = s
        attempts += 1

    # Phase 5: Final local search around best_s
    if time_left() > 2:
        for offset in range(-100, 101):
            if time_left() < 2:
                break
            s = best_s + offset
            if s < 1:
                continue
            k = find_run(s)
            if k > best_k:
                best_k = k
                best_s = s

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
