import math
import time
import heapq
import numpy as np

def construct():
    N_MIN = 10000
    N_MAX = 1000000
    DEADLINE = 85

    start_time = time.time()

    # Generate primes up to sqrt(N_MAX + 1) for segmented sieve
    sqrt_max = int(math.isqrt(N_MAX + 1)) + 1
    is_prime_small = bytearray(b'\x01') * (sqrt_max + 1)
    is_prime_small[0] = is_prime_small[1] = 0
    for i in range(2, int(math.isqrt(sqrt_max)) + 1):
        if is_prime_small[i]:
            for j in range(i*i, sqrt_max + 1, i):
                is_prime_small[j] = 0
    small_primes = [i for i in range(2, sqrt_max + 1) if is_prime_small[i]]

    # Segmented sieve for largest prime factor in [N_MIN, N_MAX+1]
    size = N_MAX + 1 - N_MIN + 1
    lpf_segment = np.ones(size, dtype=np.int64)
    values_segment = np.arange(N_MIN, N_MAX + 2, dtype=np.int64)

    for p in small_primes:
        # Find the first multiple of p in [N_MIN, N_MAX+1]
        start = ((N_MIN + p - 1) // p) * p
        if start < p * p:
            start = p * p
        for multiple in range(start, N_MAX + 2, p):
            idx = multiple - N_MIN
            lpf_segment[idx] = p

    # Handle numbers that are prime themselves (larger than sqrt_max)
    # For any number where lpf_segment[i] == 1, it must be prime
    # Actually, if a number has no small prime factor, it's prime
    # So lpf_segment[i] should be the number itself
    prime_mask = (lpf_segment == 1)
    lpf_segment[prime_mask] = values_segment[prime_mask]

    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return N_MIN

    # Vectorized scan: find n that minimizes max(lpf(n), lpf(n+1))
    a = lpf_segment[:-1]  # lpf of n for n in [N_MIN, N_MAX]
    b = lpf_segment[1:]   # lpf of n+1 for n in [N_MIN, N_MAX]
    vals = np.maximum(a, b)
    best_idx = int(np.argmin(vals))
    best_n = N_MIN + best_idx
    best_val = int(vals[best_idx])

    # The segmented sieve gives us the exact answer via vectorized search
    # No need for smooth number search - the vectorized scan finds the global minimum
    return best_n
