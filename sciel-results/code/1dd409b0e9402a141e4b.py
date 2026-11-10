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
    primes_np = np.array(primes, dtype=np.int64)

    def compute_phi_segment(start, length):
        if length <= 0:
            return []
        
        # Use int64. phi(n) <= n <= 10^12 fits in int64.
        phi = np.arange(start, start + length, dtype=np.int64)
        remaining = phi.copy()
        
        # We only need primes up to sqrt(start + length)
        # sqrt(10^12) = 10^6
        # So we iterate over all primes in primes_np
        
        for p in primes_np:
            # Optimization: if p*p > start+length, we can break?
            # No, we need to handle the case where p is a factor.
            # If p > sqrt(n), n has at most one factor p.
            # If we process all p <= sqrt(max_n), the remaining is 1 or prime.
            # So we must process all p <= 10^6.
            
            # Find first multiple of p in [start, start+length)
            first = ((start + p - 1) // p) * p
            
            if first >= start + length:
                continue

            indices = np.arange(first - start, length, p, dtype=np.int64)
            
            rem_vals = remaining[indices]
            mask = (rem_vals % p == 0)
            
            if not np.any(mask):
                continue
                
            idxs = indices[mask]
            phi[idxs] -= phi[idxs] // p
            
            vals = remaining[idxs]
            # Remove all factors of p
            # Most values are divisible by p^1.
            # A few by p^2.
            # Vectorized loop is fine.
            for _ in range(10):
                mask2 = (vals % p == 0)
                if not np.any(mask2):
                    break
                vals[mask2] //= p
            remaining[idxs] = vals

        # Handle remaining prime factors
        mask_large = remaining > 1
        if np.any(mask_large):
            idxs_large = np.where(mask_large)[0]
            phi[idxs_large] -= phi[idxs_large] // remaining[idxs_large]
        
        return phi.tolist()

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
                break
        
        return min(best_k, max_k), best_s

    best_k = 1
    best_s = 1
    
    # Baseline
    if time_left() > 10:
        k, s = find_best_run_sliding(1, 5000, 200)
        if k > best_k:
            best_k, best_s = k, s

    random.seed(42)
    
    # Search strategy:
    # Use larger segments to find longer runs.
    # Segments of 10,000 are cheap.
    
    while time_left() > 20:
        start = random.randint(10**9, 10**12 - 10000)
        length = 10000
        k, s = find_best_run_sliding(start, length, 200)
        if k > best_k:
            best_k, best_s = k, s
            
        # Also try a smaller, denser search near the found best?
        # Or just keep random searching.
        
    # Final extension
    if best_k < 200 and time_left() > 5:
        # Extend left
        left_start = max(1, best_s - 5000)
        left_len = best_s - left_start + best_k + 5000
        if left_len > 0:
            k_left, s_left = find_best_run_sliding(left_start, left_len, 200)
            if k_left > best_k:
                best_k, best_s = k_left, s_left
        
        # Extend right
        right_end = min(10**12, best_s + best_k + 5000)
        right_start = best_s
        right_len = right_end - right_start + 1
        if right_len > 0:
            k_right, s_right = find_best_run_sliding(right_start, right_len, 200)
            if k_right > best_k:
                best_k, best_s = k_right, s_right

    return {'start': best_s, 'length': best_k}
