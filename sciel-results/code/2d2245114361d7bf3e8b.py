"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    _prime_cache = {}
    
    def sieve_primes(limit):
        """Sieve primes up to limit using bytearray, with caching."""
        if limit in _prime_cache:
            return _prime_cache[limit]
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                start = i * i
                step = i
                is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
        result = [i for i in range(2, limit + 1) if is_prime[i]]
        _prime_cache[limit] = result
        return result
    
    def compute_phi_segmented(lo, hi):
        """Compute phi for numbers in [lo, hi] using segmented sieve with Python lists."""
        n = hi - lo + 1
        if n <= 0:
            return []
        limit = int(math.isqrt(hi)) + 1
        primes = sieve_primes(limit)
        
        phi = [lo + i for i in range(n)]
        remaining = [lo + i for i in range(n)]
        
        for p in primes:
            start = ((lo + p - 1) // p) * p
            for j in range(start, hi + 1, p):
                idx = j - lo
                rem = remaining[idx]
                if rem % p == 0:
                    phi[idx] -= phi[idx] // p
                    while rem % p == 0:
                        rem //= p
                    remaining[idx] = rem
        
        for i in range(n):
            rem = remaining[i]
            if rem > 1:
                phi[i] -= phi[i] // rem
        
        return phi
    
    def find_best_run(phi_arr, offset, max_len=200):
        """Find the longest run of distinct phi values in phi_arr using sliding window."""
        n = len(phi_arr)
        if n == 0:
            return offset, 0
        best_k = 0
        best_s = offset
        left = 0
        seen = {}
        
        for right in range(n):
            val = phi_arr[right]
            if val in seen:
                left = max(left, seen[val] + 1)
            seen[val] = right
            current_len = right - left + 1
            if current_len > best_k:
                best_k = min(current_len, max_len)
                best_s = offset + left
            if best_k >= max_len:
                break
        
        return best_s, best_k
    
    best_s, best_k = 1, 1
    
    # Search strategy: focus on ranges where phi values are more likely to be distinct.
    # Longer runs tend to occur in ranges with diverse prime factorizations.
    # We search progressively larger ranges, spending more time on promising areas.
    
    # Phase 1: Quick baseline scan of small ranges
    small_ranges = [
        (1, 10000),
        (10000, 30000),
        (30000, 60000),
        (60000, 120000),
    ]
    
    for lo, hi in small_ranges:
        if time_left() < 10.0:
            break
        try:
            phi_arr = compute_phi_segmented(lo, hi)
            s, k = find_best_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue
    
    # Phase 2: Medium ranges - these are where runs of 10-50 are common
    medium_ranges = [
        (100000, 150000),
        (200000, 300000),
        (500000, 700000),
        (1000000, 1500000),
        (2000000, 3000000),
        (5000000, 7000000),
        (10000000, 15000000),
        (20000000, 30000000),
        (50000000, 80000000),
    ]
    
    for lo, hi in medium_ranges:
        if time_left() < 20.0:
            break
        try:
            phi_arr = compute_phi_segmented(lo, hi)
            s, k = find_best_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue
    
    # Phase 3: Large ranges - potential for longer runs
    large_ranges = [
        (100000000, 200000000),
        (500000000, 1000000000),
        (1000000000, 2000000000),
        (5000000000, 10000000000),
        (10000000000, 20000000000),
        (50000000000, 100000000000),
        (100000000000, 200000000000),
        (500000000000, 1000000000000),
    ]
    
    for lo, hi in large_ranges:
        if time_left() < 30.0:
            break
        try:
            phi_arr = compute_phi_segmented(lo, hi)
            s, k = find_best_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue
    
    # Phase 4: Very large ranges near 10^12
    if time_left() > 40.0:
        very_large_ranges = [
            (10**11, 10**11 + 200000),
            (2 * 10**11, 2 * 10**11 + 200000),
            (5 * 10**11, 5 * 10**11 + 200000),
            (10**12 - 200000, 10**12),
        ]
        for lo, hi in very_large_ranges:
            if time_left() < 30.0:
                break
            try:
                phi_arr = compute_phi_segmented(lo, hi)
                s, k = find_best_run(phi_arr, lo, max_len=200)
                if k > best_k:
                    best_k = k
                    best_s = s
            except Exception:
                continue
    
    # Phase 5: Local refinement around best found run
    if best_k >= 5 and time_left() > 20.0:
        for delta in [-1000, -500, -200, -100, 100, 200, 500, 1000]:
            if time_left() < 10.0:
                break
            extend_lo = max(1, best_s + delta - 200)
            extend_hi = min(10**12, best_s + best_k + 200 + delta)
            if extend_hi - extend_lo < 100:
                continue
            try:
                phi_arr = compute_phi_segmented(extend_lo, extend_hi)
                s, k = find_best_run(phi_arr, extend_lo, max_len=200)
                if k > best_k:
                    best_k = k
                    best_s = s
            except Exception:
                continue
    
    # Ensure validity
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
