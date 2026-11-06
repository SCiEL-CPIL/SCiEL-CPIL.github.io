"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import isqrt
import time

def construct():
    """
    Find the longest run of consecutive integers with distinct totients up to 10^12.
    Uses segmented sieve with SPF to compute phi in O(log n) per number,
    then scans for collisions using sliding window.
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    
    MAX_N = 10**12
    MAX_K = 200
    
    # First, sieve all primes up to sqrt(10^12) = 10^6
    limit = isqrt(MAX_N)
    # Use a simple sieve for primes up to limit
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    # Function to compute phi for a range [start, end] using segmented approach
    def compute_phi_range(start, end):
        """Compute phi(n) for all n in [start, end] using segmented factorization."""
        size = end - start + 1
        # phi[n - start] will hold phi(n)
        phi_arr = np.ones(size, dtype=np.int64)
        # rem[n - start] holds the remaining part of n after removing small prime factors
        rem = np.arange(start, end + 1, dtype=np.int64)
        
        for p in primes:
            if p * p > end:
                break
            # Find first multiple of p in [start, end]
            first = ((start + p - 1) // p) * p
            if first < p * p:
                first = p * p
            for multiple in range(first, end + 1, p):
                idx = multiple - start
                if rem[idx] % p == 0:
                    phi_arr[idx] = phi_arr[idx] // p * (p - 1)
                    while rem[idx] % p == 0:
                        rem[idx] //= p
        
        # For numbers where rem > 1, there's a remaining prime factor
        for i in range(size):
            if rem[i] > 1:
                phi_arr[i] = phi_arr[i] // rem[i] * (rem[i] - 1)
        
        return phi_arr
    
    # Sliding window to find longest run of distinct phi values
    def find_longest_run(phi_arr, start_idx, max_k):
        """Find the longest run of distinct values in phi_arr using sliding window."""
        best_k = 1
        best_s = start_idx
        n = len(phi_arr)
        
        # Use a set-based sliding window
        left = 0
        freq = {}
        for right in range(n):
            val = phi_arr[right]
            freq[val] = freq.get(val, 0) + 1
            
            while freq[val] > 1:
                left_val = phi_arr[left]
                freq[left_val] -= 1
                if freq[left_val] == 0:
                    del freq[left_val]
                left += 1
            
            k = right - left + 1
            if k > best_k:
                best_k = k
                best_s = start_idx + left
            
            if best_k >= max_k:
                break
        
        return best_k, best_s
    
    # Scan through ranges, looking for longer runs
    # Start with a reasonable segment size
    segment_size = 5000000  # 5 million numbers per segment
    
    best_k = 1
    best_s = 1
    
    # First pass: scan from 1 upward with growing segments
    current_start = 1
    while current_start <= MAX_N and time.time() < deadline:
        current_end = min(current_start + segment_size - 1, MAX_N)
        
        phi_vals = compute_phi_range(current_start, current_end)
        k, s = find_longest_run(phi_vals, current_start, MAX_K)
        
        if k > best_k:
            best_k = k
            best_s = s
        
        # If we found a good run, try to extend the search around it
        if best_k >= 15:
            # Do a focused search around the best region
            focus_start = max(1, best_s - 100000)
            focus_end = min(MAX_N, best_s + best_k + 100000)
            phi_focus = compute_phi_range(focus_start, focus_end)
            k2, s2 = find_longest_run(phi_focus, focus_start, MAX_K)
            if k2 > best_k:
                best_k = k2
                best_s = s2
        
        current_start = current_end + 1
        
        # If we've been scanning for a while and haven't found improvement,
        # increase segment size to cover more ground
        if current_start > 10**7 and best_k < 10:
            segment_size = min(segment_size * 2, 20000000)
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
