"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from array import array

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Precompute primes up to a reasonable limit for segmented sieve
    # For hi up to 10^12, sqrt(hi) = 10^6
    # For smaller segments, we can use a smaller limit
    MAX_PRIME_LIMIT = 10**6 + 1
    
    def sieve_primes(limit):
        """Sieve primes up to limit using bytearray for efficiency."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i:limit+1:i] = bytearray(len(range(i*i, limit+1, i)))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Precompute primes once
    primes = sieve_primes(MAX_PRIME_LIMIT)
    
    def compute_phi_segmented(lo, hi):
        """Compute phi for numbers in [lo, hi] using segmented sieve with array module."""
        n = hi - lo + 1
        if n <= 0:
            return []
        
        # Use array('Q') for 64-bit unsigned integers (phi values can be large)
        phi = array('Q', (lo + i for i in range(n)))
        remaining = array('Q', (lo + i for i in range(n)))
        
        for p in primes:
            if p * p > hi:
                break
            start = ((lo + p - 1) // p) * p
            for j in range(start, hi + 1, p):
                idx = j - lo
                if remaining[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # For each number, if remaining > 1, it has a prime factor > sqrt(hi)
        for i in range(n):
            if remaining[i] > 1:
                phi[i] -= phi[i] // remaining[i]
        
        return phi
    
    def find_best_run(phi_arr, offset, max_len=200):
        """Find the longest run of distinct phi values in phi_arr."""
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
            i += 1
        return best_s, best_k
    
    # Strategy: We want to find a long run of consecutive integers where phi values are distinct.
    # phi(n) is even for n > 2, so consecutive phi values are both even but can be distinct.
    # Longer runs tend to occur where numbers have varied factorizations.
    
    best_s, best_k = 1, 1
    
    # First, try a range near 1 to get a baseline
    if time_left() > 5.0:
        try:
            phi_arr = compute_phi_segmented(1, 50000)
            s, k = find_best_run(phi_arr, 1, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            pass
    
    # Now do random sampling across [1, 10^12] to find longer runs
    # Key insight: longer runs of distinct totients likely occur in ranges
    # where numbers have diverse prime factorizations.
    
    # Use a deterministic random seed for reproducibility
    rng = random.Random(42)
    
    # Number of random samples to try
    num_samples = 50
    
    for _ in range(num_samples):
        if time_left() < 15.0:
            break
        
        # Choose a random start point in [1, 10^12]
        # Bias towards larger ranges where longer runs might exist
        # Use logarithmic spacing to cover the full range
        log_max = math.log10(10**12)  # = 12
        log_start = rng.uniform(0, log_max - 5)  # leave room for segment
        start = int(10 ** log_start) + 1
        start = max(1, min(start, 10**12 - 200))
        
        # Segment size: try various sizes
        # Larger segments have more chance of containing long runs
        seg_size = rng.choice([10000, 20000, 50000, 100000])
        hi = min(start + seg_size - 1, 10**12)
        
        try:
            phi_arr = compute_phi_segmented(start, hi)
            if time_left() < 10.0:
                break
            s, k = find_best_run(phi_arr, start, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue
    
    # If we've found a good run, try to extend it
    if best_k > 10 and time_left() > 20.0:
        # Try extending the found run in both directions
        for ext_size in [200, 500, 1000, 2000]:
            if time_left() < 10.0:
                break
            extend_lo = max(1, best_s - ext_size)
            extend_hi = min(10**12, best_s + best_k + ext_size)
            try:
                phi_arr = compute_phi_segmented(extend_lo, extend_hi)
                s, k = find_best_run(phi_arr, extend_lo, max_len=200)
                if k > best_k:
                    best_k = k
                    best_s = s
            except Exception:
                pass
    
    # Try some targeted ranges if we have time
    if time_left() > 30.0:
        # Ranges where numbers might have diverse factorizations
        targeted_ranges = [
            (10**9, 10**9 + 100000),
            (10**10, 10**10 + 100000),
            (10**11, 10**11 + 100000),
            (10**12 - 200000, 10**12),
        ]
        for lo, hi in targeted_ranges:
            if time_left() < 15.0:
                break
            try:
                phi_arr = compute_phi_segmented(lo, hi)
                if time_left() < 10.0:
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
