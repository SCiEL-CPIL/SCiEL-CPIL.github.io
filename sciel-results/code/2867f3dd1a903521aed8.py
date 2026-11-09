import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    # 1. Sieve primes up to 10^6 (sqrt(10^12))
    limit = 1000000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, end):
        """
        Computes phi(n) for n in [start, end] using segmented sieve.
        Returns a numpy array of phi values (int64).
        """
        size = end - start + 1
        if size <= 0:
            return np.array([], dtype=np.int64)
        
        # Use int64
        vals = np.arange(start, end + 1, dtype=np.int64)
        phis = vals.copy()
        rems = vals.copy()
        
        # Process small prime factors
        for p in primes:
            if p * p > end:
                break
            
            first_mult = ((start + p - 1) // p) * p
            
            if first_mult > end:
                continue
                
            count = (end - first_mult) // p + 1
            if count <= 0:
                continue
            
            idx_array = np.arange(first_mult - start, size, p, dtype=np.int64)
            
            # Check divisibility on remainder
            mask = rems[idx_array] % p == 0
            valid_idx = idx_array[mask]
            
            if len(valid_idx) == 0:
                continue
            
            # Update phi
            phis[valid_idx] -= phis[valid_idx] // p
            
            # Remove factors of p from rems
            # Optimization: Divide in bulk where possible, but careful with numpy broadcasting
            # Since p is small, we can do a few iterations
            for _ in range(10):
                still_div = rems[valid_idx] % p == 0
                if not np.any(still_div):
                    break
                rems[valid_idx[still_div]] //= p
                    
        # Process remaining large prime factors
        large_prime_mask = rems > 1
        if np.any(large_prime_mask):
            large_indices = np.where(large_prime_mask)[0]
            p_vals = rems[large_indices]
            phis[large_indices] -= phis[large_indices] // p_vals
            
        return phis

    def find_longest_run(phi_vals, start_index):
        """
        Finds the longest run of distinct values in phi_vals.
        phi_vals is a numpy array.
        Returns (max_length, relative_start_index).
        """
        best_k = 1
        best_s_rel = 0
        n = len(phi_vals)
        
        if n == 0:
            return 0, 0

        last_seen = {}
        current_len = 0
        current_start_rel = 0
        
        for i in range(n):
            val = int(phi_vals[i])
            
            if val in last_seen:
                last_idx = last_seen[val]
                if last_idx >= current_start_rel:
                    current_start_rel = last_idx + 1
            
            last_seen[val] = i
            current_len = i - current_start_rel + 1
            
            if current_len > best_k:
                best_k = current_len
                best_s_rel = current_start_rel
                
            if best_k >= 200:
                return best_k, best_s_rel
                
        return best_k, best_s_rel

    def check_run_distinct(start, k):
        """
        Check if phi(start), phi(start+1), ..., phi(start+k-1) are all distinct.
        """
        if start < 1 or start + k - 1 > 10**12:
            return False
        if k <= 1:
            return True
        
        phi_vals = compute_phi_segment(start, start + k - 1)
        if len(phi_vals) != k:
            return False
        
        seen = set()
        for val in phi_vals:
            v = int(val)
            if v in seen:
                return False
            seen.add(v)
        return True

    def extend_run(start, initial_k):
        """
        Try to extend a run starting at 'start' beyond initial_k.
        """
        k = initial_k
        while k < 200 and start + k <= 10**12:
            if check_run_distinct(start, k + 1):
                k += 1
            else:
                break
        return k

    best_k = 1
    best_s = 1
    
    # Strategy: Scan contiguous high ranges
    # Higher n generally means larger phi(n), which means larger gaps between phi values,
    # reducing the probability of collision in a window of size k.
    
    W = 50000  # Segment size
    
    # Define scan ranges. We want to cover as much of [5e11, 1e12] as possible.
    # We scan from high to low, as higher values are slightly better candidates for distinctness.
    scan_start_max = 10**12
    scan_start_min = 5 * 10**11
    
    # We will iterate through segments. 
    # To maximize coverage, we scan from the top down.
    current_pos = scan_start_max
    while current_pos > scan_start_min:
        if time.time() - start_time > deadline - 5:
            break
        
        seg_end = current_pos
        seg_start = max(scan_start_min, seg_end - W + 1)
        
        # Ensure we don't go below 1
        if seg_start < 1:
            seg_start = 1
            
        if seg_start > seg_end:
            break
            
        try:
            phi_vals = compute_phi_segment(seg_start, seg_end)
            k, s_rel = find_longest_run(phi_vals, seg_start)
            actual_s = seg_start + s_rel
            
            if k > best_k:
                best_k = k
                best_s = actual_s
                
                # If we find a good run, try to extend it to see if we can reach 200
                # Only do this if we have time and the run is substantial
                if best_k > 50 and time.time() - start_time < deadline - 10:
                    extended_k = extend_run(actual_s, k)
                    if extended_k > best_k:
                        best_k = extended_k
                        best_s = actual_s
                        
                if best_k >= 200:
                    return {'start': best_s, 'length': best_k}
                    
        except Exception:
            pass
            
        current_pos = seg_start - 1
        
    # If we haven't found 200, do a few random checks in very high ranges just in case
    # (Though contiguous scan should have covered most of the space)
    if time.time() - start_time < deadline - 10 and best_k < 200:
        random.seed(42)
        for _ in range(20):
            if time.time() - start_time > deadline - 5:
                break
            s = random.randint(9 * 10**11, 10**12 - W)
            try:
                phi_vals = compute_phi_segment(s, s + W - 1)
                k, s_rel = find_longest_run(phi_vals, s)
                actual_s = s + s_rel
                if k > best_k:
                    best_k = k
                    best_s = actual_s
                    if best_k > 50 and time.time() - start_time < deadline - 5:
                        extended_k = extend_run(actual_s, k)
                        if extended_k > best_k:
                            best_k = extended_k
                            best_s = actual_s
            except Exception:
                pass

    # Ensure constraints
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': best_s, 'length': best_k}
