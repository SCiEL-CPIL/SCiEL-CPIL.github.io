# EVOLVE-BLOCK-START
import time
import math
import sys

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_start = 1
    best_length = 1
    
    def compute_phi_range(start, end):
        """Compute phi(n) for n in [start, end] using segmented sieve approach."""
        n = end
        # Sieve primes up to sqrt(end)
        limit = int(math.isqrt(end)) + 1
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = False
        primes = [i for i in range(2, limit + 1) if is_prime[i]]
        
        # Initialize phi values
        phi = list(range(start, end + 1))
        remaining = list(range(start, end + 1))
        
        for p in primes:
            if p * p > end:
                break
            # Find first multiple of p in [start, end]
            first = ((start + p - 1) // p) * p
            for m in range(first, end + 1, p):
                idx = m - start
                if remaining[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors
        for i in range(len(remaining)):
            if remaining[i] > 1:
                phi[i] -= phi[i] // remaining[i]
        
        return phi
    
    def find_longest_distinct_run(phi_values, offset=0):
        """Find the longest run of consecutive distinct phi values."""
        if not phi_values:
            return 0, 0
        n = len(phi_values)
        seen = {}
        left = 0
        best_len = 0
        best_left = 0
        
        for right in range(n):
            val = phi_values[right]
            if val in seen and seen[val] >= left:
                left = seen[val] + 1
            seen[val] = right
            current_len = right - left + 1
            if current_len > best_len:
                best_len = current_len
                best_left = left
        
        return best_len, best_left + offset
    
    # First, search small ranges to find good candidates
    # Search in various strategic regions
    search_regions = [
        (1, 50000),
        (100000, 150000),
        (500000, 550000),
        (1000000, 1050000),
        (10000000, 10050000),
        (100000000, 100050000),
        (1000000000, 1000050000),
        (10000000000, 10000050000),
        (100000000000, 100000050000),
        (500000000000, 500000050000),
        (900000000000, 900000050000),
    ]
    
    for start, end in search_regions:
        if time.time() > deadline:
            break
        if end > 10**12:
            end = 10**12
        if start < 1:
            start = 1
        
        # Limit the range size for performance
        max_range = 100000
        if end - start > max_range:
            # Sample multiple sub-ranges
            step = max_range // 5
            for sub_start in range(start, end, step):
                sub_end = min(sub_start + max_range, end)
                if time.time() > deadline:
                    break
                phi_vals = compute_phi_range(sub_start, sub_end)
                length, left_offset = find_longest_distinct_run(phi_vals, sub_start)
                if length > best_length:
                    best_length = length
                    best_start = left_offset
        else:
            phi_vals = compute_phi_range(start, end)
            length, left_offset = find_longest_distinct_run(phi_vals, start)
            if length > best_length:
                best_length = length
                best_start = left_offset
    
    # If we found a good result, also try to extend it
    if best_length > 1 and time.time() < deadline - 10:
        # Try searching around the best found location for potentially longer runs
        center = best_start + best_length // 2
        search_width = max(50000, best_length * 10)
        new_start = max(1, center - search_width)
        new_end = min(10**12, center + search_width)
        
        # Search in smaller chunks
        chunk_size = 50000
        for cs in range(new_start, new_end, chunk_size):
            if time.time() > deadline:
                break
            ce = min(cs + chunk_size, new_end)
            phi_vals = compute_phi_range(cs, ce)
            length, left_offset = find_longest_distinct_run(phi_vals, cs)
            if length > best_length:
                best_length = length
                best_start = left_offset
    
    # Ensure constraints are met
    if best_start < 1:
        best_start = 1
    if best_start + best_length - 1 > 10**12:
        best_length = min(best_length, 10**12 - best_start + 1)
    if best_length < 1:
        best_length = 1
    if best_length > 200:
        best_length = 200
    
    return {'start': best_start, 'length': best_length}
# EVOLVE-BLOCK-END
