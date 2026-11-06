import time
import sys
import numpy as np
from math import isqrt

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2)."""
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
    for p in small_primes:
        pp = p * p
        if pp > temp:
            break
        if temp % pp == 0:
            while temp % p == 0:
                temp //= p
        elif temp % p == 0:
            return False
    if temp > 1:
        r = isqrt(temp)
        if r * r != temp:
            return False
    return True

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit using n = a^2 * b^3, b square-free."""
    powerful = set()
    powerful.add(1)
    max_b = int(limit ** (1.0 / 3.0)) + 2

    # Sieve of Eratosthenes up to max_b to get primes
    max_b_plus = max_b + 1
    if max_b_plus > 1:
        is_prime = bytearray(b'\x01') * (max_b_plus)
        is_prime[0] = 0
        if max_b_plus > 1:
            is_prime[1] = 0
        for i in range(2, isqrt(max_b) + 1):
            if is_prime[i]:
                for j in range(i * i, max_b_plus, i):
                    is_prime[j] = 0
        small_primes = [i for i in range(2, max_b_plus) if is_prime[i]]
    else:
        small_primes = []

    for b in range(1, max_b + 1):
        # Check if b is square-free
        t = b
        is_sf = True
        for p in small_primes:
            if p * p > t:
                break
            if t % (p * p) == 0:
                is_sf = False
                break
        if not is_sf:
            continue

        b3 = b * b * b
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit

    # Generate powerful numbers up to a limit.
    # K ~ 2.16 * sqrt(L)
    # L = 500M -> K ~ 48,300.
    pow_limit = 500_000_000  # 5 * 10^8

    try:
        powerful = _generate_powerful_numbers(pow_limit)
    except Exception:
        pow_limit = 400_000_000
        powerful = _generate_powerful_numbers(pow_limit)

    K = len(powerful)

    if time.time() > deadline - 60:
        # Fallback to smaller limit if generation was unexpectedly slow
        pow_limit = 300_000_000
        powerful = _generate_powerful_numbers(pow_limit)
        K = len(powerful)

    pow_arr = np.array(powerful, dtype=np.int64)
    pow_set = set(powerful)

    max_sum = 2 * pow_limit + 1

    try:
        counts = np.zeros(max_sum, dtype=np.int32)
    except MemoryError:
        pow_limit = 300_000_000
        powerful = _generate_powerful_numbers(pow_limit)
        K = len(powerful)
        pow_arr = np.array(powerful, dtype=np.int64)
        pow_set = set(powerful)
        max_sum = 2 * pow_limit + 1
        counts = np.zeros(max_sum, dtype=np.int32)

    # Chunked processing
    # K ~ 48,000 for 500M.
    # Chunk size 1500 -> 32 chunks.
    # Each chunk: 48,000 * 1,500 = 72,000,000 elements.
    # 32 * 72M = 2.3B elements.
    # Numpy bincount is fast.
    
    chunk_size = 1500

    for i in range(0, K, chunk_size):
        chunk = pow_arr[i:i + chunk_size]
        sums = pow_arr[:, None] + chunk[None, :]
        
        chunk_counts = np.bincount(sums.ravel(), minlength=max_sum)
        counts += chunk_counts

        if time.time() > deadline - 20:
            break

    if not any(counts > 0):
        return {'N': 5, 'pairs': [[1, 4]]}

    # Find the best N
    best_N = int(np.argmax(counts))
    best_count = int(counts[best_N])

    # Extract pairs
    pairs = []
    half_N = best_N // 2
    for a in powerful:
        if a > half_N:
            break
        b = best_N - a
        if b in pow_set:
            # Double check powerful status (should be true by construction)
            # But _is_powerful is a good safeguard if generation had bugs
            if _is_powerful(a) and _is_powerful(b):
                pairs.append([a, b])

    if not pairs:
        return {'N': 5, 'pairs': [[1, 4]]}

    return {'N': best_N, 'pairs': pairs}
