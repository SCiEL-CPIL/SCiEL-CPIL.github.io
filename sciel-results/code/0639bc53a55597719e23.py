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

def _count_pairs_for_N(N, pow_arr_small, pow_arr_large, deadline):
    """Count pairs (a, b) with a+b=N, a in pow_arr_small, b in pow_arr_large, a<=b.
    Returns list of pairs and count."""
    if time.time() > deadline - 5:
        return [], 0
    small_set = set(pow_arr_small)
    large_set = set(pow_arr_large)
    pairs = []
    half_N = N // 2
    # a must be in small, b = N-a must be in large, and a <= b => a <= half_N
    # Also b = N-a must be >= 1 and <= max(large)
    for a in pow_arr_small:
        if a > half_N:
            break
        if time.time() > deadline - 5:
            break
        b = N - a
        if b < 1:
            continue
        if b in large_set:
            if _is_powerful(a) and _is_powerful(b):
                pairs.append([a, b])
    return pairs, len(pairs)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit

    # Generate powerful numbers up to a moderate limit for coarse counting
    pow_limit_coarse = 200_000_000  # 2 * 10^8

    powerful = _generate_powerful_numbers(pow_limit_coarse)
    K = len(powerful)

    elapsed = time.time() - start_time
    if time.time() > deadline - 60:
        # Fallback to smaller limit if generation was unexpectedly slow
        pow_limit_coarse = 100_000_000
        powerful = _generate_powerful_numbers(pow_limit_coarse)
        K = len(powerful)

    pow_arr = np.array(powerful, dtype=np.int64)

    # Coarse counting: count all pairs (a,b) with a+b=N for N up to 2*pow_limit_coarse
    max_sum = 2 * pow_limit_coarse + 1
    counts = np.zeros(max_sum, dtype=np.int32)

    # Adaptive chunk size
    # Memory for one chunk: K * chunk_size * 8 bytes
    # Target ~500MB per chunk
    chunk_size = max(500, min(3000, int(500_000_000 // (K * 8))))

    for i in range(0, K, chunk_size):
        if time.time() > deadline - 30:
            break
        chunk = pow_arr[i:i + chunk_size]
        sums = pow_arr[:, None] + chunk[None, :]
        chunk_counts = np.bincount(sums.ravel(), minlength=max_sum)
        counts += chunk_counts

    if not any(counts > 0):
        return {'N': 5, 'pairs': [[1, 4]]}

    # Find top candidates from coarse count
    # Get top 10 indices by count
    top_indices = np.argsort(counts)[::-1][:10]
    top_candidates = [(int(counts[idx]), int(idx)) for idx in top_indices if counts[idx] > 0]

    if not top_candidates:
        return {'N': 5, 'pairs': [[1, 4]]}

    # Sort by count descending
    top_candidates.sort(key=lambda x: x[0], reverse=True)

    # For the top few candidates, do a targeted polish with a larger limit
    # to catch representations where one summand exceeds pow_limit_coarse
    best_pairs = []
    best_N = top_candidates[0][1]
    best_count = top_candidates[0][0]

    # Use the coarse powerful list for the "small" side
    pow_arr_small = pow_arr

    # Generate a larger powerful list for the "large" side if time permits
    pow_limit_large = 500_000_000  # 5 * 10^8
    if time.time() < deadline - 40:
        try:
            powerful_large = _generate_powerful_numbers(pow_limit_large)
            pow_arr_large = np.array(powerful_large, dtype=np.int64)
        except Exception:
            pow_arr_large = pow_arr
    else:
        pow_arr_large = pow_arr

    # Polish: for top 3 candidates, count pairs with a in small, b in large
    for count_est, N_candidate in top_candidates[:3]:
        if time.time() > deadline - 15:
            break
        pairs, count = _count_pairs_for_N(N_candidate, pow_arr_small, pow_arr_large, deadline)
        if count > len(best_pairs):
            best_pairs = pairs
            best_N = N_candidate
            best_count = count

    # If polish didn't find anything, fall back to coarse extraction
    if not best_pairs:
        pow_set = set(powerful)
        half_N = best_N // 2
        for a in powerful:
            if a > half_N:
                break
            b = best_N - a
            if b in pow_set:
                if _is_powerful(a) and _is_powerful(b):
                    best_pairs.append([a, b])

    if not best_pairs:
        return {'N': 5, 'pairs': [[1, 4]]}

    return {'N': best_N, 'pairs': best_pairs}
