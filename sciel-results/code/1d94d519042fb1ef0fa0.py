import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    # Generate primes up to 10^6 (sqrt of 10^12 is 10^6, but we only need sqrt of max R we process)
    # If we process up to 10^8, sqrt is 10^4.
    # If we process up to 10^10, sqrt is 10^5.
    # Let's generate primes up to 10^5 to be safe for ranges up to 10^10.
    limit = 10**5
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    best_k = 1
    best_s = 1
    
    def process_segment(L, R):
        nonlocal best_k, best_s
        if R > 10**12:
            R = 10**12
        if L > R:
            return
        n = R - L + 1
        # Use int64 for phi and temp
        phi = np.arange(L, R+1, dtype=np.int64)
        temp = np.arange(L, R+1, dtype=np.int64)
        
        for p in primes:
            if p * p > R:
                break
            # Find indices where temp is divisible by p
            mask = temp % p == 0
            if np.any(mask):
                phi[mask] = phi[mask] // p * (p - 1)
                temp[mask] = temp[mask] // p
        
        # Handle remaining prime factors
        mask = temp > 1
        if np.any(mask):
            q = temp[mask]
            phi[mask] = phi[mask] // q * (q - 1)
        
        # Find max run of distinct phi values
        # We need to find the longest contiguous subarray with all distinct elements.
        # Using a dictionary to track counts in a sliding window.
        counts = {}
        left = 0
        max_len = 0
        max_start = 0
        
        # Optimized loop for finding max distinct run
        # Converting to list might be faster for iteration than numpy array indexing in Python loop?
        # Or just iterate numpy array directly.
        phi_list = phi.tolist()
        
        for right in range(n):
            val = phi_list[right]
            counts[val] = counts.get(val, 0) + 1
            while counts[val] > 1:
                left_val = phi_list[left]
                counts[left_val] -= 1
                left += 1
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                max_start = left
        
        if max_len > best_k:
            best_k = max_len
            best_s = L + max_start
            
        return best_k, best_s

    # Search strategy
    # 1. Process a large initial range [1, 10^7]
    # 10^7 elements is ~80MB for two int64 arrays. Acceptable.
    if time.time() < deadline - 10:
        process_segment(1, 10**7)
        
    # 2. Try some larger, sparser ranges or specific areas if time permits
    # The maximum run is likely to be small (logarithmic).
    # We can try random sampling or specific boundaries.
    # However, the first segment [1, 10^7] is very likely to contain the global maximum or close to it
    # because the density of "bad" pairs (collisions) is uniform-ish?
    # Actually, phi(n) collisions are related to Carmichael's function etc.
    # But let's try a few more segments to be safe.
    
    if time.time() < deadline - 5:
        process_segment(10**7 + 1, 2 * 10**7)
        
    if time.time() < deadline - 2:
        # Try a segment near 10^12?
        # But we need primes up to sqrt(10^12) = 10^6.
        # Our prime list is only up to 10^5.
        # We would need to regenerate or extend primes.
        # Given the time, sticking to ranges where sqrt(R) <= 10^5 is safer/faster.
        # So R <= 10^10.
        process_segment(10**10 - 10**7, 10**10)

    return {'start': best_s, 'length': best_k}
