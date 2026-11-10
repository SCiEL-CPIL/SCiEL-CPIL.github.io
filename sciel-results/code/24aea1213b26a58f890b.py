import time
import math
import random
import sys

def construct():
    start_time = time.time()
    TIME_LIMIT = 118.0  # Safety margin before 120s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Precompute primes up to 10^6 (sqrt(10^12))
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, length):
        """
        Compute phi for all numbers in [start, start+length) using segmented sieve.
        Returns list of phi values.
        """
        if length <= 0:
            return []
        
        # Initialize phi and remaining factors
        # Using lists of integers for flexibility, though numpy could be faster for large lengths
        # However, standard lists are fine for length ~100k-1M given the operations
        phi = list(range(start, start + length))
        rem = list(range(start, start + length))
        
        # Process each prime
        for p in primes:
            if p >= start + length:
                break
            
            # Find first multiple of p in [start, start+length)
            first = ((start + p - 1) // p) * p
            if first < start:
                first += p
            
            # Iterate through multiples
            for j in range(first, start + length, p):
                idx = j - start
                if rem[idx] % p == 0:
                    # Remove all factors of p from rem[idx]
                    while rem[idx] % p == 0:
                        rem[idx] //= p
                    # Update phi
                    phi[idx] -= phi[idx] // p
        
        # Handle remaining prime factors > sqrt(10^12)
        for i in range(length):
            if rem[i] > 1:
                phi[i] -= phi[i] // rem[i]
        
        return phi

    def find_best_run_in_segment(phis, base_start, max_k=200):
        """
        Find the longest run of distinct phi values in the given list.
        Returns (best_k, best_start_offset).
        """
        n = len(phis)
        if n == 0:
            return 0, 0
            
        best_k = 1
        best_offset = 0
        left = 0
        freq = {}
        
        for right in range(n):
            val = phis[right]
            freq[val] = freq.get(val, 0) + 1
            
            # Shrink window from left until all values are distinct
            while freq[val] > 1:
                left_val = phis[left]
                freq[left_val] -= 1
                if freq[left_val] == 0:
                    del freq[left_val]
                left += 1
            
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_offset = left
                
            # Early exit if we hit the max possible k
            if best_k >= max_k:
                return max_k, best_offset
        
        return best_k, best_offset

    # Search Parameters
    # We search in segments of size SEG_SIZE
    # Larger SEG_SIZE allows finding longer runs but takes more time per segment
    # 200,000 is a good balance. It takes ~1-2 seconds to compute phi for this range.
    SEG_SIZE = 200000 
    MAX_K = 200
    
    best_k = 1
    best_s = 1
    
    # Initialize random seed for reproducibility if needed, but random is fine
    random.seed(42)
    
    # First, do a quick check in small numbers to ensure baseline
    if time_left() > 10:
        phis_small = compute_phi_segment(1, 10000)
        k, off = find_best_run_in_segment(phis_small, 1, MAX_K)
        if k > best_k:
            best_k = k
            best_s = 1 + off

    # Main search loop
    # We pick random start points in [10^6, 10^12]
    # To avoid overlapping segments wasting time, we could track visited, 
    # but random uniform distribution makes overlaps rare enough for 120s.
    
    while time_left() > 5:
        # Choose a random start point
        # Bias towards larger numbers? The density of distinct totients might vary.
        # Uniform random in log space or linear space?
        # Let's try uniform linear in [10^6, 10^12]
        # Actually, runs of distinct totients are more likely in "smooth" regions? 
        # Or is it uniform?
        # Let's stick to uniform random sampling of segment starts.
        
        # To ensure we cover the space well, we can pick a random point 
        # and center a segment around it, or just pick a random start.
        # Picking a random start in [1, 10^12 - SEG_SIZE]
        
        # Optimization: If we find a good run, we might want to explore neighbors.
        # But random sampling is robust.
        
        start = random.randint(10**6, 10**12 - SEG_SIZE)
        
        # Compute phi for the segment
        phis = compute_phi_segment(start, SEG_SIZE)
        
        # Find best run in this segment
        k, off = find_best_run_in_segment(phis, start, MAX_K)
        
        if k > best_k:
            best_k = k
            best_s = start + off
            
            # If we found a very long run, we might want to check if it can be extended
            # But the sliding window already finds the max distinct run within the segment.
            # If the run hits the boundary of the segment, we should extend the search.
            # Check if best_s is near the start or end of the segment
            if best_s < start + 1000 and time_left() > 10:
                # Extend left
                new_start = max(1, start - SEG_SIZE)
                new_len = start - new_start + SEG_SIZE
                phis_ext = compute_phi_segment(new_start, new_len)
                k_ext, off_ext = find_best_run_in_segment(phis_ext, new_start, MAX_K)
                if k_ext > best_k:
                    best_k = k_ext
                    best_s = new_start + off_ext
                    
            if best_s + best_k > start + SEG_SIZE - 1000 and time_left() > 10:
                # Extend right
                new_end = min(10**12, start + SEG_SIZE + SEG_SIZE)
                new_len = new_end - (start + SEG_SIZE)
                if new_len > 0:
                    phis_ext = compute_phi_segment(start + SEG_SIZE, new_len)
                    k_ext, off_ext = find_best_run_in_segment(phis_ext, start + SEG_SIZE, MAX_K)
                    if k_ext > best_k:
                        best_k = k_ext
                        best_s = start + SEG_SIZE + off_ext

    return {'start': best_s, 'length': best_k}

if __name__ == "__main__":
    # For local testing
    res = construct()
    print(res)
