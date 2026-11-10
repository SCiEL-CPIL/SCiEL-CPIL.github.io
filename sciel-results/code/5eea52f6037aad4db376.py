import sys
import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # We will search for k up to N
    # Let's start with a reasonable N and increase if time permits
    # But we can just pick a large N that fits in time.
    # N = 10^7 is safe.
    N = 10_000_000
    
    # Sieve for sigma
    sigma = np.ones(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if sigma[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                p_power = i
                term = 1
                while j % p_power == 0:
                    term += p_power
                    p_power *= i
                    if p_power > N:
                        break
                # Wait, this is not efficient.
                # Standard sieve for sigma:
                # sigma(n) = product (p^(a+1)-1)/(p-1)
                # We can use a modified sieve.
                pass
    
    # Better sieve for sigma:
    # sigma = [1] * (N+1)
    # for i in range(2, N+1):
    #     if sigma[i] == 1: # prime
    #         for j in range(i, N+1, i):
    #             p_power = i
    #             term = 1
    #             while j % p_power == 0:
    #                 term += p_power
    #                 p_power *= i
    #             sigma[j] *= term
    # This is O(N log N) roughly.
    
    # Let's use a more efficient linear sieve or standard sieve.
    # Since N=10^7, O(N log N) is ~ 10^7 * 16 = 1.6 * 10^8 operations.
    # In Python, this might be too slow (10-20s).
    # Let's optimize or use a smaller N.
    # N = 2 * 10^6 might be faster.
    
    # Let's try N = 2 * 10^6 first.
    N = 2_000_000
    
    sigma = np.ones(N + 1, dtype=np.int64)
    # Standard sieve for sigma
    for i in range(2, N + 1):
        if sigma[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                p_power = i
                term = 1
                while j % p_power == 0:
                    term += p_power
                    p_power *= i
                sigma[j] *= term
    
    # Check time
    if time.time() - start_time > deadline - 10:
         # Fallback to small hardcoded solution if sieve is too slow
         return {'M': 333312, 'k': [336, 372, 434]}

    # Compute M_k and count
    counts = {}
    for k in range(1, N + 1):
        M_val = k * sigma[k]
        if M_val not in counts:
            counts[M_val] = []
        counts[M_val].append(k)
        
    # Find max
    best_M = 0
    best_k_list = []
    for M_val, k_list in counts.items():
        if len(k_list) > len(best_k_list):
            best_M = M_val
            best_k_list = k_list
            
    # Ensure types are Python int
    best_M = int(best_M)
    best_k_list = [int(k) for k in best_k_list]
    
    # Verify constraint k <= sqrt(M)
    # As discussed, k < sqrt(M) for k>1.
    # For k=1, M=2, 1 <= sqrt(2).
    
    return {'M': best_M, 'k': best_k_list}
