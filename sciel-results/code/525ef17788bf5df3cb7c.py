import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 118.0  # safety margin before 120s kill

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
        Optimized with vectorized operations where possible.
        """
        if length <= 0:
            return np.array([], dtype=np.int64)
        
        # Initialize phi values to the numbers themselves
        phi_vals = np.arange(start, start + length, dtype=np.int64)
        # rem tracks the remaining part of the number to detect large prime factors
        rem = np.arange(start, start + length, dtype=np.int64)
        
        # Process small primes - only need primes up to max(start+length-1)
        # But we can limit to primes where p <= start+length-1
        max_val = start + length - 1
        
        for p in primes:
            if p > max_val:
                break
            
            # Find first multiple of p in [start, start+length)
            first = ((start + p - 1) // p) * p
            if first >= start + length:
                continue
            
            # Create indices of multiples
            indices = np.arange(first, start + length, p, dtype=np.int64) - start
            if len(indices) == 0:
                continue
            
            # Apply phi factor: phi(n) = n * (1 - 1/p) for each prime factor p
            # phi_vals[indices] = phi_vals[indices] // p * (p - 1)
            phi_vals[indices] = (phi_vals[indices] // p) * (p - 1)
            
            # Remove all factors of p from rem
            # Vectorized: repeatedly divide rem[indices] by p while divisible
            # For efficiency, do it in chunks
            for _ in range(10):  # Max exponent for p^k <= 10^12 is about 40 for p=2, but 10 is safe
                mask = rem[indices] % p == 0
                if not np.any(mask):
                    break
                rem[indices[mask]] //= p
        
        # Handle large prime factors
        # If rem[i] > 1, it is a prime factor > max(primes)
        mask = rem > 1
        if np.any(mask):
            phi_vals[mask] = (phi_vals[mask] // rem[mask]) * (rem[mask] - 1)
            
        return phi_vals

    def find_best_run_sliding(start, length, max_k=200):
        """
        Find the longest run of distinct phi values in [start, start+length)
        """
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
    
    # Baseline check
    if time_left() > 10:
        k, s = find_best_run_sliding(1, 1000, 200)
        if k > best_k:
            best_k, best_s = k, s

    random.seed(42)
    
    # Adaptive search: start with smaller segments, increase if time allows
    # Use a mix of strategies: random sampling + local refinement
    seg_len = 1000  # Start smaller for more iterations
    
    while time_left() > 5:
        # Random sampling
        start = random.randint(10**6, 10**12 - seg_len)
        k, s = find_best_run_sliding(start, seg_len, 200)
        if k > best_k:
            best_k, best_s = k, s
            # If we found a long run, try to extend it
            if best_k > 10 and time_left() > 10:
                # Extend left and right
                ext_start = max(1, best_s - 500)
                ext_len = best_s - ext_start + best_k + 500
                k_ext, s_ext = find_best_run_sliding(ext_start, ext_len, 200)
                if k_ext > best_k:
                    best_k, best_s = k_ext, s_ext
                
                ext_start_r = best_s
                ext_end_r = min(10**12, best_s + best_k + 500)
                ext_len_r = ext_end_r - ext_start_r + 1
                k_ext_r, s_ext_r = find_best_run_sliding(ext_start_r, ext_len_r, 200)
                if k_ext_r > best_k:
                    best_k, best_s = k_ext_r, s_ext_r
        
        # Also try some deterministic spots: powers of 2, factorials, etc.
        if time_left() > 20:
            # Try around highly composite numbers
            for base in [10**6, 10**7, 10**8, 10**9, 10**10, 10**11]:
                if time_left() < 5:
                    break
                start_det = base - seg_len // 2
                if start_det < 1:
                    start_det = 1
                k_det, s_det = find_best_run_sliding(start_det, seg_len, 200)
                if k_det > best_k:
                    best_k, best_s = k_det, s_det
    
    return {'start': best_s, 'length': best_k}
