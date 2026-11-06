"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    def time_remaining():
        return DEADLINE - (time.time() - start_time)
    
    # Known maximal runs from literature/OEIS for consecutive integers with distinct totients
    # These are verified runs where phi(s), phi(s+1), ..., phi(s+k-1) are all pairwise distinct
    # Format: (start, length)
    # For large ranges up to 10^12, known results suggest runs of length ~10-15 are common
    # but longer runs are rare. Let's use a combination of known values and search.
    
    # First, let's try to find good runs using a sieve-based approach for phi computation
    # For small numbers, we can compute phi efficiently using a sieve
    
    def compute_phi_sieve(limit):
        """Compute phi(n) for all n up to limit using a sieve."""
        phi = np.arange(limit + 1, dtype=np.int64)
        for i in range(2, limit + 1):
            if phi[i] == i:  # i is prime
                for j in range(i, limit + 1, i):
                    phi[j] -= phi[j] // i
        return phi
    
    def find_longest_distinct_run(phi_values, max_start, max_length=200):
        """Find the longest run of consecutive values with distinct totients."""
        best_k = 1
        best_s = 1
        n = len(phi_values) - 1  # phi_values is 1-indexed, so n = limit
        
        for s in range(1, min(max_start, n - max_length + 2)):
            seen = set()
            k = 0
            while s + k <= n and phi_values[s + k] not in seen:
                seen.add(phi_values[s + k])
                k += 1
            if k > best_k:
                best_k = k
                best_s = s
            if k >= max_length:
                break
        
        return best_s, best_k
    
    # Strategy 1: Compute phi for a moderate range and find the best run
    # Let's try a range that's large enough to find good runs but small enough to compute quickly
    
    # For the range up to 10^12, we can't compute phi for all numbers.
    # But we can search in strategic ranges.
    
    # Known from literature: runs of length ~10-15 are common.
    # Let's try to find runs in various ranges.
    
    best_overall_k = 1
    best_overall_s = 1
    
    # Try computing phi for a moderate range first
    # Let's try up to 1,000,000 first
    limit1 = 1000000
    if time_remaining() > 10:
        phi_small = compute_phi_sieve(limit1)
        s1, k1 = find_longest_distinct_run(phi_small, limit1 - 200)
        if k1 > best_overall_k:
            best_overall_k = k1
            best_overall_s = s1
    
    # Try a larger range if time permits
    limit2 = 5000000
    if time_remaining() > 20:
        phi_mid = compute_phi_sieve(limit2)
        s2, k2 = find_longest_distinct_run(phi_mid, limit2 - 200)
        if k2 > best_overall_k:
            best_overall_k = k2
            best_overall_s = s2
    
    # For very large numbers, we need a different approach
    # We can use the fact that phi(n) is even for n > 2, and odd only for n = 1, 2
    # So for n > 2, phi(n) is always even
    
    # Let's try to find runs in larger ranges by sampling
    # For large n, we can use trial division for phi computation
    
    def phi_large(n):
        """Compute phi(n) for a single large number using trial division."""
        if n == 1:
            return 1
        result = n
        d = 2
        while d * d <= n:
            if n % d == 0:
                while n % d == 0:
                    n //= d
                result -= result // d
            d += 1 if d == 2 else 2  # Skip even numbers after 2
        if n > 1:
            result -= result // n
        return result
    
    # Try to find runs in larger ranges by checking specific positions
    # We'll sample various starting points and check if we can extend a run
    
    # For large numbers, let's try ranges around powers of 2 or other structured points
    # where we might find longer runs of distinct totients
    
    # Try ranges around 10^6, 10^7, 10^8, etc.
    search_ranges = [
        (1000000, 1000000 + 10000),
        (10000000, 10000000 + 10000),
        (100000000, 100000000 + 10000),
        (1000000000, 1000000000 + 10000),
        (10000000000, 10000000000 + 10000),
        (100000000000, 100000000000 + 10000),
    ]
    
    for base, size in search_ranges:
        if time_remaining() < 5:
            break
        if base + size > 10**12:
            continue
        
        # Try to find a run starting near base
        # We'll check a window of starting positions
        window = 1000
        for s in range(base, min(base + window, 10**12 - 200)):
            if time_remaining() < 5:
                break
            
            # Try to extend a run from s
            seen = set()
            k = 0
            valid = True
            for i in range(200):
                if s + i > 10**12:
                    break
                val = phi_large(s + i)
                if val in seen:
                    break
                seen.add(val)
                k += 1
            
            if k > best_overall_k:
                best_overall_k = k
                best_overall_s = s
    
    # Return the best result found
    return {'start': best_overall_s, 'length': best_overall_k}
# EVOLVE-BLOCK-END
