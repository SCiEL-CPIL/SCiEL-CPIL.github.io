"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import isqrt
import time

def construct():
    """
    Find the longest run of consecutive integers with distinct totients up to 10^12.
    Uses a segmented sieve to compute phi values efficiently.
    Scans a large contiguous range starting from 1 to maximize the chance of finding long runs.
    """
    MAX_N = 10**12
    MAX_K = 200
    
    # Precompute primes up to sqrt(MAX_N) = 10^6
    limit = isqrt(MAX_N) + 1
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    small_primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def compute_phi_segment(start, end):
        """
        Compute phi(n) for all n in [start, end] using segmented sieve.
        """
        rem = np.arange(start, end + 1, dtype=np.int64)
        phi_vals = np.arange(start, end + 1, dtype=np.int64)
        
        for p in small_primes:
            if p * p > end:
                break
            first = ((start + p - 1) // p) * p
            if first > end:
                continue
            indices = np.arange(first, end + 1, p) - start
            phi_vals[indices] = (phi_vals[indices] // p) * (p - 1)
            rem[indices] = rem[indices] // p
            while True:
                mask = (rem[indices] % p == 0)
                if not np.any(mask):
                    break
                rem[indices[mask]] = rem[indices[mask]] // p
        
        large_prime_mask = rem > 1
        if np.any(large_prime_mask):
            q = rem[large_prime_mask]
            phi_vals[large_prime_mask] = (phi_vals[large_prime_mask] // q) * (q - 1)
        
        if start == 1:
            phi_vals[0] = 1
            
        return phi_vals

    def find_longest_run(phi_arr):
        """
        Find the longest contiguous subarray with all distinct values.
        """
        n = len(phi_arr)
        best_len = 1
        best_start = 0
        last_seen = {}
        left = 0
        
        for right in range(n):
            val = phi_arr[right]
            if val in last_seen and last_seen[val] >= left:
                left = last_seen[val] + 1
            last_seen[val] = right
            
            curr_len = right - left + 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = left
        
        return best_start, best_len

    start_time = time.time()
    TIME_LIMIT = 115  # Leave 5s buffer
    
    best_k = 1
    best_s = 1
    
    # Scan from 1 upwards with a moderate segment size to balance overhead
    SEG_SIZE = 2 * 10**6
    seg_start = 1
    
    while seg_start <= MAX_N and (time.time() - start_time) < TIME_LIMIT:
        seg_end = min(seg_start + SEG_SIZE - 1, MAX_N)
        phi_seg = compute_phi_segment(seg_start, seg_end)
        local_start, local_len = find_longest_run(phi_seg)
        
        if local_len > best_k:
            best_k = local_len
            best_s = seg_start + local_start
        
        seg_start = seg_end + 1
        
        # Optional: If we've scanned a very large range, we can break early if time is tight
        # but the while condition handles this.
        
    best_k = min(best_k, MAX_K)
    
    # Verification
    if best_k > 1:
        verify_end = best_s + best_k - 1
        if verify_end <= MAX_N:
            phi_verify = compute_phi_segment(best_s, verify_end)
            seen = set()
            valid = True
            for v in phi_verify:
                if v in seen:
                    valid = False
                    break
                seen.add(v)
            if not valid:
                _, valid_len = find_longest_run(phi_verify)
                best_k = min(best_k, valid_len)
                
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
