import time
import math
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 145.0
    
    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # 1. Sieve primes up to 10^6
    LIMIT_SQRT = 10**6
    is_prime = bytearray(b'\x01') * (LIMIT_SQRT + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(LIMIT_SQRT)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, LIMIT_SQRT + 1) if is_prime[i]]

    # 2. Function to compute phi for a block [L, R] using NumPy
    def compute_phi_block(L, R):
        n = R - L + 1
        phi = np.arange(L, R + 1, dtype=np.int64)
        
        for p in primes:
            if p > R:
                break
            # Find first multiple of p in [L, R]
            m = ((L + p - 1) // p) * p
            start_idx = m - L
            if start_idx >= n:
                continue
            # Vectorized update
            indices = np.arange(start_idx, n, p)
            phi[indices] = phi[indices] // p * (p - 1)
        return phi

    # 3. Function to find longest distinct run in a list of values
    # Converting numpy array to list for set operations might be slow.
    # But we can work with the numpy array directly?
    # Set operations on numpy arrays are not efficient.
    # Converting to list: list(phi) takes time.
    # Alternative: Use a dictionary or set on the Python list.
    # For 1e6 elements, list conversion is ~0.05s.
    # Let's stick to Python list for the scan, it's robust.
    
    def find_longest_distinct_run(vals, base_offset):
        # vals is a numpy array
        # Convert to list for faster set operations?
        # Actually, iterating over numpy array is slower than list.
        # So convert to list.
        v_list = vals.tolist()
        
        seen = set()
        left = 0
        max_len = 0
        best_left = 0
        
        for right in range(len(v_list)):
            val = v_list[right]
            while val in seen:
                seen.remove(v_list[left])
                left += 1
            seen.add(val)
            
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                best_left = left
        
        if max_len == 0:
            return 0, base_offset
        return max_len, base_offset + best_left

    best_k = 1
    best_s = 1
    
    # 4. Search strategy with overlap
    BLOCK_SIZE = 2_000_000 # Larger blocks with NumPy
    OVERLAP = 200
    STEP = BLOCK_SIZE - OVERLAP
    
    current_R = 10**12
    
    while current_R >= BLOCK_SIZE and time_left() > 5:
        L = current_R - BLOCK_SIZE + 1
        if L < 1:
            L = 1
            
        phi_vals = compute_phi_block(L, current_R)
        
        k, s = find_longest_distinct_run(phi_vals, L)
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k == 200:
                break
        
        current_R -= STEP
        
        if current_R < 1:
            break

    return {'start': best_s, 'length': best_k}
