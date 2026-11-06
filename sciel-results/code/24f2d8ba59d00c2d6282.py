"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import sys
from array import array

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 150s kill
    
    def now():
        return time.time()
    
    def elapsed():
        return now() - start_time
    
    # Segmented sieve to compute phi for all numbers in [lo, hi]
    def phi_segmented(lo, hi):
        """Compute phi(n) for all n in [lo, hi] using segmented sieve."""
        size = hi - lo + 1
        if size <= 0:
            return []
        
        # Initialize phi values to n (identity)
        phi_vals = array('Q', [i + lo for i in range(size)])
        
        # Find all primes up to sqrt(hi)
        limit = int(math.isqrt(hi)) + 1
        is_prime = [True] * (limit + 1)
        if limit >= 0:
            is_prime[0] = False
        if limit >= 1:
            is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        
        primes = [i for i in range(2, limit + 1) if is_prime[i]]
        
        # For each prime p, update phi for multiples of p in [lo, hi]
        for p in primes:
            # Find first multiple of p in [lo, hi]
            first = ((lo + p - 1) // p) * p
            for m in range(first, hi + 1, p):
                idx = m - lo
                if phi_vals[idx] % p == 0:
                    phi_vals[idx] = phi_vals[idx] // p * (p - 1)
        
        return phi_vals
    
    # Search in a range [lo, hi] for longest run with distinct phi values
    def search_range(lo, hi, max_window=200):
        nonlocal best_start, best_len
        
        if hi - lo + 1 < 2:
            return
        
        # Compute phi for the entire range
        phi_vals = phi_segmented(lo, hi)
        n_vals = len(phi_vals)
        
        if n_vals < 2:
            return
        
        # Slide a window to find longest run with distinct phi values
        # Use a set to track seen phi values
        seen = set()
        left = 0
        
        for right in range(n_vals):
            if now() > DEADLINE:
                return
            
            val = phi_vals[right]
            
            # If val is already in seen, we need to shrink from left
            while val in seen and left <= right:
                seen.remove(phi_vals[left])
                left += 1
            
            seen.add(val)
            window_len = right - left + 1
            
            if window_len > best_len:
                best_len = window_len
                best_start = lo + left
    
    best_start = 1
    best_len = 1
    
    # Strategy: search in regions where numbers have many distinct prime factors
    # These are more likely to have distinct totients
    
    # Phase 1: Small ranges (can search up to ~10^5)
    for hi in [100, 1000, 10000, 100000]:
        if now() > DEADLINE:
            break
        search_range(1, hi, max_window=200)
    
    # Phase 2: Medium ranges - focus on regions with many composites
    if now() < DEADLINE:
        # Search around highly composite numbers
        targets = [5040, 25200, 50400, 100800, 252000, 504000, 1008000, 5040000, 10080000]
        for t in targets:
            if now() > DEADLINE:
                break
            lo = max(1, t - 50000)
            hi = t + 50000
            if hi > 10**12:
                hi = 10**12
            search_range(lo, hi, max_window=200)
    
    # Phase 3: Larger ranges in chunks - search in arithmetic progressions
    if now() < DEADLINE:
        # Search in blocks around numbers with many small prime factors
        chunk_size = 20000
        bases = [10**6, 10**7, 10**8, 10**9, 10**10, 10**11, 10**12 // 2]
        for base in bases:
            if now() > DEADLINE:
                break
            # Search a window around each base
            lo = max(1, base - chunk_size)
            hi = base + chunk_size
            if hi > 10**12:
                hi = 10**12
            if lo < 1:
                lo = 1
            search_range(lo, hi, max_window=200)
            
            # Also search at base + offset for different offsets
            for offset in [chunk_size * 2, chunk_size * 5, chunk_size * 10]:
                if now() > DEADLINE:
                    break
                lo2 = base + offset
                hi2 = lo2 + chunk_size
                if hi2 > 10**12:
                    hi2 = 10**12
                if lo2 >= hi2 or lo2 > 10**12:
                    continue
                search_range(lo2, hi2, max_window=200)
    
    # Phase 4: If time remains, do broader random sampling
    if now() < DEADLINE - 5:
        import random
        random.seed(42)
        # Try some random large ranges
        for _ in range(20):
            if now() > DEADLINE:
                break
            # Pick a random start point
            lo = random.randint(10**6, 10**12 - 100000)
            hi = lo + 50000
            if hi > 10**12:
                hi = 10**12
            if lo < 1:
                lo = 1
            search_range(lo, hi, max_window=200)
    
    # Final check
    if best_len < 1:
        best_len = 1
        best_start = 1
    
    return {'start': best_start, 'length': best_len}
# EVOLVE-BLOCK-END
