import numpy as np
import time

def construct():
    N = 10**7
    
    # 1. Precompute sigma using a sieve
    # sigma[i] will hold the sum of divisors of i
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    # 2. Search for best pair
    best_a = 1
    best_b = 1
    best_err = float('inf')
    
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    # We want to minimize |a/b - 1.5| = k / (2b) where k = |2a - 3b|
    # Iterate k starting from 1
    
    # We will check k = 1, 2, 3, ...
    # For each k, we want the largest b such that sigma(a)==sigma(b)
    # and a = (3b +/- k) / 2 is an integer in [1, N]
    
    for k in range(1, 20000):
        if time.time() > deadline:
            break
            
        # Check if further k can possibly improve
        # min_error_for_k = k / (2 * N)
        if best_err < k / (2.0 * N):
            break
            
        # Determine parity constraint
        # a = (3b + k) / 2 or (3b - k) / 2
        # 3b +/- k must be even => b and k must have same parity
        
        parity_k = k % 2
        # b must have same parity as k
        # Start from largest b <= N with correct parity
        if N % 2 == parity_k:
            b_start = N
        else:
            b_start = N - 1
            
        # Calculate minimum b required to improve best_err
        # k / (2 * b) < best_err  =>  b > k / (2 * best_err)
        if best_err > 0:
            b_min = int(k / (2.0 * best_err)) + 1
        else:
            b_min = 1
            
        if b_start < b_min:
            continue
            
        # Case 1: 2a = 3b + k => a = (3b + k) / 2
        # a <= N => 3b + k <= 2N => b <= (2N - k) / 3
        b_max_case1 = int((2 * N - k) / 3)
        if b_max_case1 < b_min:
            pass
        else:
            # Iterate b downwards from min(b_start, b_max_case1) to b_min
            b_upper = min(b_start, b_max_case1)
            # Adjust b_upper to correct parity if needed (should already be correct from b_start logic)
            # But if we clamped with b_max_case1, check parity
            if b_upper % 2 != parity_k:
                b_upper -= 1
                
            for b in range(b_upper, b_min - 1, -2):
                a = (3 * b + k) // 2
                if a <= N and a >= 1:
                    if sigma[a] == sigma[b]:
                        err = k / (2.0 * b)
                        if err < best_err:
                            best_err = err
                            best_a = a
                            best_b = b
                        break # Found largest b for this k and case, move to next k or other case?
                        # Actually, we should check Case 2 as well for the same k?
                        # Case 2 might have a larger b?
                        # b_max_case2 = int((2N + k) / 3) which is larger than b_max_case1.
                        # So Case 2 allows larger b.
                        # We should check Case 2 first if it allows larger b?
                        # Or just check both and keep the best b?
                        # Since we iterate b downwards, the first match is the largest b for that case.
                        # We should compare the b from Case 1 and Case 2.
                        # Let's restructure: check both cases, find max b, then update.
                        
        # Let's restart the inner logic to handle both cases properly.
        # We want the global max b for this k.
        
        # Re-doing the search for this k properly
        b_found = 0
        a_found = 0
        
        # Case 1: a = (3b + k) / 2
        b_max1 = int((2 * N - k) / 3)
        b_upper1 = min(b_start, b_max1)
        if b_upper1 % 2 != parity_k:
            b_upper1 -= 1
        if b_upper1 >= b_min:
            for b in range(b_upper1, b_min - 1, -2):
                a = (3 * b + k) // 2
                if sigma[a] == sigma[b]:
                    b_found = b
                    a_found = a
                    break
        
        # Case 2: a = (3b - k) / 2
        # a >= 1 => 3b - k >= 2 => b >= (k+2)/3
        b_min2 = int((k + 2) / 3) + 1
        if b_min2 < b_min:
            b_min2 = b_min
        # b <= N
        b_upper2 = b_start
        if b_upper2 % 2 != parity_k:
            b_upper2 -= 1
        if b_upper2 >= b_min2:
            for b in range(b_upper2, b_min2 - 1, -2):
                a = (3 * b - k) // 2
                if a >= 1 and sigma[a] == sigma[b]:
                    # We want the largest b. 
                    # If we already found a match in Case 1, compare b.
                    if b_found == 0 or b > b_found:
                        b_found = b
                        a_found = a
                    break # Largest b for Case 2 found
                
        if b_found > 0:
            err = k / (2.0 * b_found)
            if err < best_err:
                best_err = err
                best_a = a_found
                best_b = b_found
                
    # Fallback
    if best_a == 1 and best_b == 1:
        best_a = 11
        best_b = 6
        
    return [int(best_a), int(best_b)]
