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
    
    def sieve_primes(limit):
        """Sieve primes up to limit using bytearray."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                start = i * i
                step = i
                is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    def compute_phi_segmented_numpy(lo, hi):
        """Compute phi for numbers in [lo, hi] using NumPy for speed."""
        n = hi - lo + 1
        if n <= 0:
            return np.array([], dtype=np.int64)
        limit = int(math.isqrt(hi)) + 1
        primes = sieve_primes(limit)
        
        # Use Python int for phi values to avoid overflow issues with large numbers
        phi = np.array([lo + i for i in range(n)], dtype=object)
        remaining = np.array([lo + i for i in range(n)], dtype=object)
        
        for p in primes:
            start = ((lo + p - 1) // p) * p
            for j in range(start, hi + 1, p):
                idx = j - lo
                rem = remaining[idx]
                if rem % p == 0:
                    phi_val = phi[idx]
                    phi[idx] = phi_val - phi_val // p
                    while rem % p == 0:
                        rem = rem // p
                    remaining[idx] = rem
        
        # For each number, if remaining > 1, it has a prime factor > limit
        for i in range(n):
            rem = remaining[i]
            if rem > 1:
                phi_val = phi[i]
                phi[i] = phi_val - phi_val // rem
        
        return phi.tolist()
    
    def compute_phi_segmented_fast(lo, hi):
        """Optimized phi computation for smaller ranges."""
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
            # If we've hit max_len, we can stop early
            if best_k >= max_len:
                break
        
        return best_s, best_k
    
    best_s, best_k = 1, 1
    
    # Strategy: Search in ranges where phi collisions are less frequent.
    # Numbers with many distinct prime factors tend to have more "spread" in phi values.
    # We'll focus on ranges around primorial-like numbers and larger ranges.
    
    # First, quick scan of small ranges to establish baseline
    small_ranges = [
        (1, 5000),
        (5000, 10000),
        (10000, 20000),
        (20000, 40000),
        (40000, 80000),
        (80000, 160000),
    ]
    
    for lo, hi in small_ranges:
        if time_left() < 15.0:
            break
        try:
            phi_arr = compute_phi_segmented_fast(lo, hi)
            if time_left() < 10.0:
                break
            s, k = find_best_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue
    
    # Now search in larger, more targeted ranges
    # Focus on ranges where numbers have varied prime factorizations
    large_ranges = [
        (200000, 220000),
        (500000, 520000),
        (1000000, 1020000),
        (5000000, 5020000),
        (10000000, 10020000),
        (50000000, 50020000),
        (100000000, 100020000),
        (500000000, 500020000),
        (1000000000, 1000020000),
        (5000000000, 5000020000),
        (10000000000, 10000020000),
        (50000000000, 50000020000),
        (100000000000, 100000020000),
        (500000000000, 500000020000),
        (1000000000000, 1000000020000),
    ]
    
    for lo, hi in large_ranges:
        if time_left() < 20.0:
            break
        try:
            if hi - lo > 100000:
                phi_arr = compute_phi_segmented_numpy(lo, hi)
            else:
                phi_arr = compute_phi_segmented_fast(lo, hi)
            if time_left() < 15.0:
                break
            s, k = find_best_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue
    
    # If we found a good run, try to extend it by searching nearby
    if best_k >= 10 and time_left() > 25.0:
        for delta in [-500, -200, -100, 100, 200, 500]:
            if time_left() < 15.0:
                break
            extend_lo = max(1, best_s + delta - 100)
            extend_hi = min(10**12, best_s + best_k + 100 + delta)
            if extend_hi - extend_lo < 50:
                continue
            try:
                if extend_hi - extend_lo > 100000:
                    phi_arr = compute_phi_segmented_numpy(extend_lo, extend_hi)
                else:
                    phi_arr = compute_phi_segmented_fast(extend_lo, extend_hi)
                if time_left() < 10.0:
                    break
                s, k = find_best_run(phi_arr, extend_lo, max_len=200)
                if k > best_k:
                    best_k = k
                    best_s = s
            except Exception:
                continue
    
    # Final targeted search in very large ranges if time permits
    if time_left() > 40.0:
        very_large_ranges = [
            (10**11, 10**11 + 50000),
            (2 * 10**11, 2 * 10**11 + 50000),
            (5 * 10**11, 5 * 10**11 + 50000),
            (10**12 - 100000, 10**12),
        ]
        for lo, hi in very_large_ranges:
            if time_left() < 25.0:
                break
            try:
                phi_arr = compute_phi_segmented_numpy(lo, hi)
                if time_left() < 20.0:
                    break
                s, k = find_best_run(phi_arr, lo, max_len=200)
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
