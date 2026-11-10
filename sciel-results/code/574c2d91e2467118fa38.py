"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def sieve_primes(limit):
        """Sieve of Eratosthenes to find all primes up to limit."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    def compute_phi_segmented(lo, hi, primes):
        """
        Compute phi(n) for all n in [lo, hi] using segmented approach.
        Returns a list of phi values.
        """
        n = hi - lo + 1
        if n <= 0:
            return []
        
        # Initialize phi and remaining arrays
        phi = list(range(lo, hi + 1))
        remaining = list(range(lo, hi + 1))
        
        # Handle phi(1) = 1 if lo == 1
        if lo == 1:
            phi[0] = 1
            remaining[0] = 1
        
        # For each prime p, update phi and remaining for multiples of p
        for p in primes:
            if p > hi:
                break
            # Find the first multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            for j in range(start, hi + 1, p):
                idx = j - lo
                if remaining[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors > sqrt(hi)
        for i in range(n):
            if remaining[i] > 1:
                phi[i] -= phi[i] // remaining[i]
        
        return phi
    
    def find_best_run(phi_arr, offset, max_len=200):
        """
        Find the longest run of distinct phi values in phi_arr.
        Returns (start_index_in_original, length).
        """
        n = len(phi_arr)
        best_k = 0
        best_s = offset
        
        # Use a sliding window with early termination
        for i in range(n):
            if n - i < best_k + 1:
                break  # Can't beat best_k with remaining elements
            seen = set()
            k = 0
            while i + k < n and k < max_len:
                val = phi_arr[i + k]
                if val in seen:
                    break
                seen.add(val)
                k += 1
            if k > best_k:
                best_k = k
                best_s = offset + i
        
        return best_s, best_k
    
    # Precompute primes up to 10^6 (enough for sqrt(10^12))
    # This is the main cost; do it once
    max_prime_limit = 10**6
    primes = sieve_primes(max_prime_limit)
    
    best_s, best_k = 1, 1
    
    # Strategy: Focus on larger numbers where phi values are more spread out
    # and thus more likely to be distinct in a run
    
    # Phase 1: Try structured ranges at different scales
    # Larger numbers have larger phi values, so collisions are less likely
    test_ranges = [
        (10**6, 10**6 + 50000),
        (10**7, 10**7 + 50000),
        (10**8, 10**8 + 50000),
        (10**9, 10**9 + 50000),
        (10**10, 10**10 + 50000),
        (10**11, 10**11 + 50000),
        (10**12 - 100000, 10**12),
    ]
    
    for lo, hi in test_ranges:
        if time_left() < 5.0:
            break
        try:
            # Use primes up to sqrt(hi), capped at our precomputed limit
            prime_limit = min(int(math.isqrt(hi)) + 1, max_prime_limit)
            phi_arr = compute_phi_segmented(lo, hi, primes)
            if time_left() < 3.0:
                break
            s, k = find_best_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue
    
    # Phase 2: Random search in upper ranges
    # Longer runs are more likely for larger numbers
    random.seed(42)
    max_n = 10**12
    
    # Try random segments, biased toward larger values
    # Use log-uniform sampling to explore different scales
    for _ in range(100):
        if time_left() < 15.0:
            break
        
        # Bias toward larger numbers: sample log10(lo) uniformly in [6, 12]
        log_lo = random.uniform(6, 12)
        lo = int(10**log_lo)
        
        seg_size = random.randint(20000, 100000)
        hi = lo + seg_size - 1
        
        if hi > max_n:
            hi = max_n
            lo = max(1, hi - seg_size + 1)
        
        if lo < 1:
            lo = 1
            hi = min(max_n, lo + seg_size - 1)
        
        if lo >= hi:
            continue
        
        try:
            prime_limit = min(int(math.isqrt(hi)) + 1, max_prime_limit)
            phi_arr = compute_phi_segmented(lo, hi, primes)
            if time_left() < 5.0:
                break
            s, k = find_best_run(phi_arr, lo, max_len=200)
            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            continue
    
    # Phase 3: If we found a good run, try to extend it
    if best_k > 10 and time_left() > 20.0:
        # Try extending the best run found by searching nearby regions
        for delta in range(-5000, 5001, 500):
            if time_left() < 10.0:
                break
            new_lo = best_s + delta
            new_hi = new_lo + 2000
            if new_lo < 1 or new_hi > max_n:
                continue
            try:
                prime_limit = min(int(math.isqrt(new_hi)) + 1, max_prime_limit)
                phi_arr = compute_phi_segmented(new_lo, new_hi, primes)
                s, k = find_best_run(phi_arr, new_lo, max_len=200)
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
    if best_s < 1:
        best_s = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
