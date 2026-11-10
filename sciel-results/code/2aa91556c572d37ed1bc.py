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

    # Sieve for primes up to limit
    def sieve_primes(lim):
        is_prime = bytearray(b'\x01') * (lim + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(lim)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, lim + 1) if is_prime[i]]

    primes = sieve_primes(10**6)

    def phi_fast(n):
        result = n
        m = n
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    best = (1, 1)

    # Phase 1: Linear sieve for phi up to 1,000,000
    end1 = 1000000
    phi_arr = [0] * (end1 + 1)
    phi_arr[1] = 1
    is_comp = bytearray(end1 + 1)
    primes_list = []
    for i in range(2, end1 + 1):
        if not is_comp[i]:
            primes_list.append(i)
            phi_arr[i] = i - 1
        for p in primes_list:
            prod = i * p
            if prod > end1:
                break
            is_comp[prod] = 1
            if i % p == 0:
                phi_arr[prod] = phi_arr[i] * p
                break
            else:
                phi_arr[prod] = phi_arr[i] * (p - 1)

    # Scan for longest run
    for s in range(1, end1 - 200):
        seen = set()
        k = 0
        while s + k <= end1 and phi_arr[s + k] not in seen:
            seen.add(phi_arr[s + k])
            k += 1
        if k > best[0]:
            best = (k, s)
        if time_left() < 50:
            break

    # Phase 2: Random sampling in [1, 10^12]
    random.seed(42)
    while time_left() > 5:
        s = random.randint(1, 10**12 - 200)
        seen = set()
        k = 0
        for i in range(200):
            v = phi_fast(s + i)
            if v in seen:
                break
            seen.add(v)
            k += 1
        if k > best[0]:
            best = (k, s)

    return {'start': best[1], 'length': best[0]}
# EVOLVE-BLOCK-END
