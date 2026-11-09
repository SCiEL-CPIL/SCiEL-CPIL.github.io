import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Increase N due to NumPy speedup
    N = 60000000
    
    # 1. Compute Euler's Totient function phi(n) for n in [1, N]
    # Using NumPy for vectorized sieve
    phi = np.arange(N + 1, dtype=np.int64)
    
    for i in range(2, N + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
            
    # Check time after sieve
    if time.time() - start_time > deadline - 10:
        return {'start': 1, 'length': 1}

    # 2. Find the longest run of consecutive integers with distinct totients
    # Sliding window
    counts = [0] * (N + 1)
    
    best_k = 1
    best_s = 1
    
    l = 1
    r = 1
    
    for r in range(1, N + 1):
        val = phi[r]
        
        if counts[val] > 0:
            while l < r:
                left_val = phi[l]
                counts[left_val] -= 1
                l += 1
                if left_val == val:
                    break
        
        counts[val] += 1
        
        current_k = r - l + 1
        if current_k > best_k:
            best_k = current_k
            best_s = l
            
        if best_k >= 200:
            break
            
        if time.time() - start_time > deadline - 5:
            break

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
