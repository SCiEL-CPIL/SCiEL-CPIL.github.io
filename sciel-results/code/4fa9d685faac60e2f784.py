"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def compute_phi_segmented(lo, hi):
        """Compute phi for numbers in [lo, hi] using segmented approach."""
        n = hi - lo + 1
        if n <= 0:
            return []
        
        # Sieve primes up to sqrt(hi)
        limit = int(math.isqrt(hi)) + 1
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        primes = []
        for i in range(2, limit + 1):
            if is_prime[i]:
                primes.append(i)
                if i * i <= limit:
                    for j in range(i * i, limit + 1, i):
                        is_prime[j] = 0
        
        # phi[i] will store the value for number lo+i
        phi = list(range(lo, hi + 1))
        remaining = list(range(lo, hi + 1))
        
        for p in primes:
            start = ((lo + p - 1) // p) * p
            for j in range(start, hi + 1, p):
                idx = j - lo
                if remaining[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        for i in range(n):
            if remaining[i] > 1:
                phi[i] -= phi[i] // remaining[i]
        
        return phi

    def find_longest_distinct_run(phi_arr, offset, max_len=200):
        """Find the longest run of distinct phi values."""
        n = len(phi_arr)
        best_k = 0
        best_s = offset
        i = 0
        while i < n:
            seen = set()
            k = 0
            while i + k < n and k < max_len and phi_arr[i + k] not in seen:
                seen.add(phi_arr[i + k])
                k += 1
            if k > best_k:
                best_k = k
                best_s = offset + i
            if k < max_len:
                i += 1
            else:
                i += k
        return best_s, best_k

    # Strategy: Search multiple ranges at different scales
    best_s, best_k = 1, 1
    
    # Ranges to test - mix of small dense ranges and larger sparse ones
    test_ranges = [
        (1, 50000),
        (100000, 200000),
        (1000000, 1200000),
        (5000000, 5200000),
        (20000000, 20200000),
        (100000000, 100200000),
        (500000000, 500200000),
        (2000000000, 2000200000),
        (10000000000, 10000200000),
        (100000000000, 100000200000),
        (500000000000, 500000200000),
        (999999999900, 999999999999),
    ]
    
    # Add some randomized ranges for diversity
    random.seed(42)
    for _ in range(5):
        scale = random.choice([10**7, 10**8, 10**9, 10**10, 10**11])
        lo = random.randint(1, max(1, 10**12 - scale - 100000))
        hi = min(10**12, lo + random.randint(50000, 200000))
        test_ranges.append((lo, hi))
    
    for lo, hi in test_ranges:
        if time_left() < 10.0:
            break
        if hi > 10**12:
            hi = 10**12
        if lo > hi or lo < 1:
            continue
        
        try:
            phi_arr = compute_phi_segmented(lo, hi)
            if time_left() < 5.0:
                break
            
            s, k = find_longest_distinct_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue
    
    # If we haven't found a very long run, do a more thorough search
    if best_k < 50 and time_left() > 20.0:
        # Search in a denser way around promising regions
        for base in [10**6, 10**7, 10**8, 10**9]:
            if time_left() < 10.0:
                break
            for offset in range(0, 10 * base, base // 10 if base > 10 else 1):
                if time_left() < 5.0:
                    break
                lo = base + offset
                hi = min(10**12, lo + 50000)
                if lo < 1 or hi > 10**12:
                    continue
                try:
                    phi_arr = compute_phi_segmented(lo, hi)
                    s, k = find_longest_distinct_run(phi_arr, lo, max_len=200)
                    if k > best_k:
                        best_k = k
                        best_s = s
                except Exception:
                    continue
    
    # Final safety checks
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
        best_k = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
