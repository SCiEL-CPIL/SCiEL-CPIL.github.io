import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 115.0  # Safety margin

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Step 1: Sieve primes up to 10^6
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
        Optimized for speed.
        """
        if length <= 0:
            return np.array([], dtype=np.int64)
        
        phi_vals = np.arange(start, start + length, dtype=np.int64)
        rem = np.arange(start, start + length, dtype=np.int64)
        
        max_val = start + length - 1
        
        # Only primes p such that p <= max_val are needed for factorization
        # But for the segmented sieve approach, we iterate all primes up to sqrt(max_val) 
        # or up to max_val? 
        # Actually, if p > sqrt(max_val), it can appear at most once.
        # The standard segmented sieve for phi uses primes up to sqrt(max_val) 
        # and then handles the remainder.
        # However, if we use primes up to max_val, we handle everything in the loop.
        # But there are ~78k primes up to 10^6.
        # For each prime, we do vector ops.
        # This is acceptable.
        
        for p in primes:
            if p > max_val:
                break
            
            # Find first multiple of p in [start, start+length)
            first = ((start + p - 1) // p) * p
            if first >= start + length:
                continue
            
            # Create indices of multiples
            # We can avoid creating a full arange if we use slicing?
            # indices = np.arange(first, start + length, p, dtype=np.int64) - start
            # This is O(count).
            
            # Alternative: Use boolean mask? 
            # mask = np.zeros(length, dtype=bool)
            # mask[first-start::p] = True
            # This might be faster for small p, but for large p, count is small.
            
            indices = np.arange(first, start + length, p, dtype=np.int64) - start
            if len(indices) == 0:
                continue
            
            # Apply phi factor
            phi_vals[indices] = (phi_vals[indices] // p) * (p - 1)
            
            # Remove all factors of p from rem
            # We need to be careful with the loop here.
            # Instead of a while loop with np.any, we can do:
            # while True:
            #    mask = rem[indices] % p == 0
            #    if not np.any(mask): break
            #    rem[indices[mask]] //= p
            
            # Optimization: Since p is prime, and we are dividing rem,
            # the number of times we can divide is small (log_p(rem)).
            # For p=2, it's up to 40. For large p, it's 1.
            
            current_indices = indices
            while True:
                mask = rem[current_indices] % p == 0
                if not np.any(mask):
                    break
                rem[current_indices[mask]] //= p
                # The indices that were divisible are still at the same positions in the original array?
                # No, we are modifying rem in place.
                # The next iteration checks the same indices again.
                # This is correct.
                # But we can filter indices to only those that were divisible?
                # No, because some might become divisible again? No, if we divide by p, 
                # and it was divisible, it might still be divisible by p.
                # So we must check the same set of indices again.
                # However, we can just keep checking the original indices.
                # The mask will become false for those that are no longer divisible.
        
        # Handle large prime factors
        mask = rem > 1
        if np.any(mask):
            phi_vals[mask] = (phi_vals[mask] // rem[mask]) * (rem[mask] - 1)
            
        return phi_vals

    def find_best_run_sliding(start, length, max_k=200):
        if length <= 0:
            return 0, start
        
        phis = compute_phi_segment(start, length)
        
        best_k = 1
        best_s = start
        left = 0
        freq = {}
        
        for right in range(length):
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
                best_s = start + left
                if best_k >= max_k:
                    return max_k, best_s
        
        return min(best_k, max_k), best_s

    best_k = 1
    best_s = 1
    
    random.seed(42)
    
    # Strategy: 
    # 1. Try a few structured high-n points.
    # 2. Then random search.
    # We want to maximize the number of segments checked.
    # Let's use segment length 20,000.
    # 120s / 20,000 numbers. 
    # If 1 segment takes 0.5s, we can do 240 segments.
    # Let's aim for 100 segments.
    
    seg_len = 20000
    max_segments = 100
    
    # Pre-defined "good" spots (optional, random is usually better)
    # Let's just do random.
    
    for i in range(max_segments):
        if time_left() < 5:
            break
        if best_k >= 200:
            break
            
        # Random start in [10^9, 10^12 - seg_len]
        # Uniform distribution
        min_start = 10**9
        max_start = 10**12 - seg_len
        if max_start < min_start:
            start = min_start
        else:
            start = random.randint(min_start, max_start)
            
        k, s = find_best_run_sliding(start, seg_len, 200)
        
        # Try to extend the run found
        if k > 1:
            # Check left extension
            ext_start = max(1, s - 100)
            ext_len = s - ext_start + k + 100
            # Only if we have time and it's worth it
            if time_left() > 2:
                k_ext, s_ext = find_best_run_sliding(ext_start, ext_len, 200)
                if k_ext > k:
                    k, s = k_ext, s_ext
            
        if k > best_k:
            best_k, best_s = k, s
            if best_k >= 200:
                break

    # If we still have time, try smaller segments in case we missed a dense cluster?
    # No, larger segments are better for finding long runs.
    
    return {'start': best_s, 'length': best_k}
