import sys
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
MAX_N = 10000000

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Compute sigma(n) for all n <= MAX_N using NumPy
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Sieve
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d
        
    # Convert to Python list for faster access in the search loop?
    # numpy scalar access is decent, but list might be faster.
    # Let's try numpy first. If too slow, convert.
    # Actually, converting 10^7 ints to list takes ~0.5s.
    # Accessing list is ~50ns. Accessing numpy array is ~100ns?
    # Let's convert.
    sigma_list = sigma.tolist()
    
    # 2. Search for best pair (a, b)
    best_a, best_b = 1, 1 # Fallback, will be updated
    best_err = float('inf')
    
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # Let k = |2a - 3b|.
    # We iterate b from MAX_N down to 1.
    
    for b in range(MAX_N, 0, -1):
        # Pruning: If the minimum possible error for this b is already >= best_err, break.
        # Min error is 1/(2b) (for odd b, k=1) or 1/b (for even b, k=2).
        # 1/(2b) is the global lower bound for any b.
        if 1.0 / (2 * b) >= best_err:
            break
            
        sb = sigma_list[b]
        
        if b % 2 == 1:
            # b is odd. 3b is odd.
            # 2a - 3b must be odd for integer a.
            # So k = |2a - 3b| must be odd.
            # Min k = 1.
            # Candidates for k=1:
            # 2a = 3b - 1 => a = (3b - 1) / 2
            # 2a = 3b + 1 => a = (3b + 1) / 2
            
            a1 = (3 * b - 1) // 2
            a2 = (3 * b + 1) // 2
            
            found_k1 = False
            if 1 <= a1 <= MAX_N and sigma_list[a1] == sb:
                # Found k=1. This is the global optimum.
                return [int(a1), int(b)]
            if 1 <= a2 <= MAX_N and sigma_list[a2] == sb:
                # Found k=1. This is the global optimum.
                return [int(a2), int(b)]
            
            # If no k=1, check k=3, 5, ... to update best_err
            # We only need to find the best k for this b to update best_err.
            # Since we iterate k upwards, the first match is the best for this b.
            for k in [3, 5, 7, 9, 11, 13]:
                a1 = (3 * b - k) // 2
                a2 = (3 * b + k) // 2
                
                if 1 <= a1 <= MAX_N and sigma_list[a1] == sb:
                    err = k / (2.0 * b)
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a1, b
                    break
                if 1 <= a2 <= MAX_N and sigma_list[a2] == sb:
                    err = k / (2.0 * b)
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a2, b
                    break
        else:
            # b is even. 3b is even.
            # 2a - 3b must be even.
            # So k = |2a - 3b| must be even.
            # Min k = 2.
            # Candidates for k=2:
            # 2a = 3b - 2 => a = (3b - 2) / 2 = 3b/2 - 1
            # 2a = 3b + 2 => a = (3b + 2) / 2 = 3b/2 + 1
            
            a1 = (3 * b - 2) // 2
            a2 = (3 * b + 2) // 2
            
            if 1 <= a1 <= MAX_N and sigma_list[a1] == sb:
                err = 1.0 / b # 2 / (2b)
                if err < best_err:
                    best_err = err
                    best_a, best_b = a1, b
                # No need to check further for this b, as k=2 is min.
            elif 1 <= a2 <= MAX_N and sigma_list[a2] == sb:
                err = 1.0 / b
                if err < best_err:
                    best_err = err
                    best_a, best_b = a2, b
            else:
                # Check k=4, 6, ...
                for k in [4, 6, 8, 10, 12, 14]:
                    a1 = (3 * b - k) // 2
                    a2 = (3 * b + k) // 2
                    
                    if 1 <= a1 <= MAX_N and sigma_list[a1] == sb:
                        err = k / (2.0 * b)
                        if err < best_err:
                            best_err = err
                            best_a, best_b = a1, b
                        break
                    if 1 <= a2 <= MAX_N and sigma_list[a2] == sb:
                        err = k / (2.0 * b)
                        if err < best_err:
                            best_err = err
                            best_a, best_b = a2, b
                        break
        
        # Time check
        if time.time() > deadline:
            break
            
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
