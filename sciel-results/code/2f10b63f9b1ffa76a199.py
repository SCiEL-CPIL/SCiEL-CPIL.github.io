import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Increase N using int32 to save memory
    N = 100000000
    
    # 1. Compute Euler's Totient function phi(n) for n in [1, N]
    # Using NumPy for vectorized sieve
    phi = np.arange(N + 1, dtype=np.int32)
    
    # Efficient prime sieve to get list of primes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    limit = int(np.sqrt(N)) + 1
    for i in range(2, limit):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = np.nonzero(is_prime)[0]
    
    # Apply totient updates for each prime
    for p in primes:
        phi[p::p] -= phi[p::p] // p
        
    # Check time after sieve
    if time.time() - start_time > deadline - 10:
        return {'start': 1, 'length': 1}

    # 2. Find the longest run of consecutive integers with distinct totients
    # Sliding window with dictionary for counts to save memory
    counts = {}
    
    best_k = 1
    best_s = 1
    
    l = 1
    r = 1
    
    phi_arr = phi
    
    for r in range(1, N + 1):
        val = phi_arr[r]
        
        if val in counts:
            while l < r:
                left_val = phi_arr[l]
                counts[left_val] -= 1
                if counts[left_val] == 0:
                    del counts[left_val]
                l += 1
                if left_val == val:
                    break
        
        if val in counts:
            counts[val] += 1
        else:
            counts[val] = 1
        
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
