import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 118.0  # safety margin

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Sieve primes up to 10^6
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, length):
        """
        Compute phi(x) for x in [start, start+length-1] using segmented sieve.
        """
        if length <= 0:
            return np.array([], dtype=np.int64)
        
        phi_vals = np.arange(start, start + length, dtype=np.int64)
        rem = np.arange(start, start + length, dtype=np.int64)
        
        max_val = start + length - 1
        
        for p in primes:
            if p > max_val:
                break
            
            first = ((start + p - 1) // p) * p
            if first >= start + length:
                continue
            
            indices = np.arange(first, start + length, p, dtype=np.int64) - start
            if len(indices) == 0:
                continue
            
            phi_vals[indices] = (phi_vals[indices] // p) * (p - 1)
            
            # Remove factors of p from rem
            while True:
                mask = rem[indices] % p == 0
                if not np.any(mask):
                    break
                rem[indices[mask]] //= p
        
        mask = rem > 1
        if np.any(mask):
            phi_vals[mask] = (phi_vals[mask] // rem[mask]) * (rem[mask] - 1)
            
        return phi_vals

    def find_best_run_in_phis(phis, max_k=200):
        """
        Find the longest run of distinct values in phis array.
        Returns (length, offset_in_phis)
        """
        if len(phis) == 0:
            return 0, 0
        
        best_k = 1
        best_idx = 0
        left = 0
        freq = {}
        
        for right in range(len(phis)):
            val = phis[right]
            freq[val] = freq.get(val, 0) + 1
            
            while freq[val] > 1:
                left_val = phis[left]
                freq[left_val] -= 1
                if freq[left_val] == 0:
                    del freq[left_val]
                left += 1
            
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_idx = left
            if best_k >= max_k:
                break
                
        return best_k, best_idx

    best_k = 1
    best_s = 1
    
    # Strategy 1: Sequential scan from 1 to 10^9
    # This is likely to find the global max if it's small enough.
    chunk_size = 10**6
    max_sequential = 10**9
    
    for start in range(1, max_sequential, chunk_size):
        if time_left() < 10:
            break
        
        end = min(start + chunk_size, max_sequential)
        length = end - start
        phis = compute_phi_segment(start, length)
        
        k, offset = find_best_run_in_phis(phis, 200)
        if k > best_k:
            best_k = k
            best_s = start + offset
            # If we found a very long run, we can try to extend it
            if best_k > 50 and time_left() > 30:
                # Extend left
                ext_start = max(1, best_s - 1000)
                ext_len = best_s - ext_start + best_k + 1000
                if ext_len > 0:
                    ext_phis = compute_phi_segment(ext_start, ext_len)
                    ext_k, ext_off = find_best_run_in_phis(ext_phis, 200)
                    if ext_k > best_k:
                        best_k = ext_k
                        best_s = ext_start + ext_off
                        
                # Extend right
                ext_start_r = best_s
                ext_end_r = min(10**12, best_s + best_k + 1000)
                ext_len_r = ext_end_r - ext_start_r + 1
                if ext_len_r > 0:
                    ext_phis_r = compute_phi_segment(ext_start_r, ext_len_r)
                    ext_k_r, ext_off_r = find_best_run_in_phis(ext_phis_r, 200)
                    if ext_k_r > best_k:
                        best_k = ext_k_r
                        best_s = ext_start_r + ext_off_r

    # Strategy 2: Random sampling in [10^9, 10^12]
    random.seed(42)
    while time_left() > 5:
        start = random.randint(10**9, 10**12 - chunk_size)
        phis = compute_phi_segment(start, chunk_size)
        k, offset = find_best_run_in_phis(phis, 200)
        if k > best_k:
            best_k = k
            best_s = start + offset
            # Local refinement
            if best_k > 20 and time_left() > 10:
                ext_start = max(1, best_s - 500)
                ext_len = best_s - ext_start + best_k + 500
                if ext_len > 0:
                    ext_phis = compute_phi_segment(ext_start, ext_len)
                    ext_k, ext_off = find_best_run_in_phis(ext_phis, 200)
                    if ext_k > best_k:
                        best_k = ext_k
                        best_s = ext_start + ext_off

    return {'start': best_s, 'length': best_k}
