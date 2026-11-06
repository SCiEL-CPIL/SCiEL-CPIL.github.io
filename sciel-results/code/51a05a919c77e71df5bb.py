"""Find longest run of consecutive integers with distinct totients up to 10^12."""
# EVOLVE-BLOCK-START
import numpy as np
from math import isqrt
import time

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    MAX_N = 10**12
    MAX_K = 200
    
    # Precompute primes up to sqrt(10^12) = 10^6
    # Actually we need primes up to 10^6 for trial division
    limit = 10**6 + 1
    sieve = bytearray(b'\x01') * limit
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, limit) if sieve[i]]
    
    def phi(x):
        """Compute Euler's totient of x using precomputed primes."""
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    def find_best_run(lo, hi):
        """Find the longest run of consecutive integers in [lo, hi) with distinct phi values."""
        if lo >= hi or hi - lo < 2:
            return (1, lo)
        
        n = hi - lo
        phi_vals = np.empty(n, dtype=np.int64)
        for i in range(n):
            phi_vals[i] = phi(lo + i)
        
        # Find longest run with all distinct values
        best_k = 1
        best_s = lo
        
        # Use a sliding window approach with a set
        # For efficiency, we'll scan and track the last occurrence of each phi value
        last_seen = {}
        window_start = 0
        
        for i in range(n):
            v = phi_vals[i]
            if v in last_seen and last_seen[v] >= window_start:
                window_start = last_seen[v] + 1
            last_seen[v] = i
            k = i - window_start + 1
            if k > best_k:
                best_k = k
                best_s = lo + window_start
        
        return (best_k, best_s)
    
    # Strategy 1: Search in the range [1, 200000] thoroughly
    # This is a good range where collisions are less frequent
    if time.time() - start_time > DEADLINE:
        return {'start': 1, 'length': 1}
    
    best_k, best_s = find_best_run(1, 200001)
    
    # Strategy 2: Try larger ranges with segmented approach
    # For large ranges, we can't compute all phi values at once
    # Instead, we use a windowed approach
    
    def find_best_run_windowed(lo, hi, window_size=50000):
        """Find best run in [lo, hi) using a sliding window."""
        if lo >= hi:
            return (1, lo)
        
        best_k = 1
        best_s = lo
        
        # We'll process in chunks and maintain a sliding window of phi values
        # Use a dictionary to track the last position of each phi value
        last_pos = {}
        window_lo = lo
        
        i = lo
        while i < hi:
            chunk_end = min(i + window_size, hi)
            for j in range(i, chunk_end):
                v = phi(j)
                if v in last_pos and last_pos[v] >= window_lo:
                    # Need to advance window_lo
                    window_lo = last_pos[v] + 1
                last_pos[v] = j
                k = j - window_lo + 1
                if k > best_k:
                    best_k = k
                    best_s = window_lo
            i = chunk_end
            
            # Clean up old entries from last_pos to save memory
            if len(last_pos) > 1000000:
                # Remove entries that are too far back
                cutoff = window_lo - 1
                keys_to_remove = [k for k in last_pos if last_pos[k] < cutoff]
                for k in keys_to_remove:
                    del last_pos[k]
        
        return (best_k, best_s)
    
    # Try several ranges
    ranges_to_try = [
        (1, 500000),
        (10**6, 10**6 + 200000),
        (10**8, 10**8 + 200000),
        (10**10, 10**10 + 200000),
    ]
    
    for lo, hi in ranges_to_try:
        if time.time() - start_time > DEADLINE:
            break
        if hi > MAX_N + 1:
            hi = MAX_N + 1
        k, s = find_best_run_windowed(lo, hi)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Strategy 3: Random sampling in large ranges
    import random
    random.seed(42)
    
    for _ in range(50):
        if time.time() - start_time > DEADLINE:
            break
        # Pick a random starting point
        lo = random.randint(1, MAX_N - 100000)
        hi = min(lo + 100000, MAX_N + 1)
        k, s = find_best_run_windowed(lo, hi, window_size=10000)
        if k > best_k:
            best_k = k
            best_s = s
    
    # Strategy 4: Targeted search around numbers with many small prime factors
    # These are more likely to have distinct phi values in their neighborhood
    if time.time() - start_time > DEADLINE:
        pass
    else:
        # Try starts that are products of first few primes
        def primorial(n):
            result = 1
            for p in primes:
                result *= p
                if result > MAX_N:
                    break
            return result
        
        # Try near primorials
        for start in [6, 30, 210, 2310, 30030, 510510, 9699690, 223092870]:
            if time.time() - start_time > DEADLINE:
                break
            if start + 100000 > MAX_N:
                continue
            lo = max(1, start - 50000)
            hi = min(start + 50000, MAX_N + 1)
            k, s = find_best_run_windowed(lo, hi, window_size=10000)
            if k > best_k:
                best_k = k
                best_s = s
    
    # Ensure constraints are met
    if best_s < 1:
        best_s = 1
    if best_s + best_k - 1 > MAX_N:
        best_k = max(1, MAX_N - best_s + 1)
    if best_k > MAX_K:
        best_k = MAX_K
    if best_k < 1:
        best_k = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
