import time
import sys
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
MAX_N = 10_000_000
DEADLINE_SECONDS = 85  # Safety margin before 90s kill

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE_SECONDS - (time.time() - t0)
        
    N = MAX_N
    
    # Step 1: Compute sigma(n) for all n up to N using NumPy sieve
    sigma_np = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve of Eratosthenes variant for sigma
    # Complexity: O(N log N)
    # NumPy vectorization makes this fast enough for N=10^7
    for i in range(1, N + 1):
        if time_left() < 10:
            # Fallback if sigma computation is too slow
            return [11, 6]
        sigma_np[i::i] += i
        
    if time_left() < 5:
        return [11, 6]
        
    best_err = float('inf')
    best_pair = (11, 6) # Fallback: sigma(11)=12, sigma(6)=12, err=|11/6-1.5|=|1.833-1.5|=0.333
    
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b).
    # Let k = |2a - 3b|. Then a = (3b +/- k)/2.
    # For a to be integer, 3b and k must have same parity.
    # If k is odd, b must be even.
    # If k is even, b must be odd.
    
    # Search strategy:
    # Iterate k from 1 upwards.
    # For each k, iterate b from N downwards.
    # Check if sigma(a) == sigma(b).
    # Stop k loop if k / (2N) >= best_err.
    
    # How large can k get?
    # If best_err is small, k limit is small.
    # If we don't find matches, k might need to be large.
    # But sigma collisions are frequent.
    # Let's set a hard limit on k, say 100000, but break earlier if possible.
    
    K_LIMIT = 100000
    
    for k in range(1, K_LIMIT + 1):
        # Check if minimum possible error for this k is better than current best
        # min_err_for_k = k / (2 * N)
        if k / (2.0 * N) >= best_err:
            break
            
        if time_left() < 2:
            break
            
        # Determine parity of b
        # 3b + k must be even => b and k same parity
        # 3b - k must be even => b and k same parity
        # So b must have the same parity as k.
        
        # Start b from N, adjust to match parity of k
        b_start = N
        if (b_start % 2) != (k % 2):
            b_start -= 1
            
        # Iterate b downwards
        # We can stop early if k / (2 * b) >= best_err
        # b_min_for_this_k = ceil(k / (2 * best_err))
        b_min = int(np.ceil(k / (2.0 * best_err))) if best_err > 0 else 1
        if b_min < 1:
            b_min = 1
            
        # b goes from b_start down to b_min
        # Step is 2 to maintain parity
        
        # We can vectorize this check partially?
        # b_arr = np.arange(b_start, b_min - 1, -2)
        # But we need to check sigma for two a values.
        # a1 = (3*b + k)//2
        # a2 = (3*b - k)//2
        
        # Let's try vectorized search for each k
        # This might be faster than a Python loop for b
        
        if b_start < b_min:
            continue
            
        b_arr = np.arange(b_start, b_min - 1, -2, dtype=np.int64)
        
        # Calculate a1 and a2
        a1 = (3 * b_arr + k) // 2
        a2 = (3 * b_arr - k) // 2
        
        # Check bounds for a1
        valid1 = (a1 >= 1) & (a1 <= N) & (a1 != b_arr)
        # Check bounds for a2
        valid2 = (a2 >= 1) & (a2 <= N) & (a2 != b_arr)
        
        found_any = False
        
        if np.any(valid1):
            a1_valid = a1[valid1]
            b_valid = b_arr[valid1]
            # Check sigma match
            matches = sigma_np[a1_valid] == sigma_np[b_valid]
            if np.any(matches):
                # We found matches. The first one in the array (largest b) is the best for this k.
                # But we need the global best.
                # Since we iterate k from 1 up, and b from N down,
                # the first match found in the entire process (smallest k, largest b) is the global best?
                # No. k=1 might match at b=100. err=1/200=0.005.
                # k=2 might match at b=9000000. err=2/18000000=1.1e-7.
                # So we must compare errors.
                
                # For this k, the best b is the largest b among matches.
                # b_arr is sorted descending. So the first True in `matches` corresponds to the largest b.
                idx = np.argmax(matches)
                b_best_k = int(b_valid[idx])
                a_best_k = int(a1_valid[idx])
                
                err = k / (2.0 * b_best_k)
                if err < best_err:
                    best_err = err
                    best_pair = (a_best_k, b_best_k)
                    found_any = True
                    # We can break out of the vectorized check?
                    # We have the best for this k. No need to check others for this k.
                    # But we still need to check a2?
                    # a2 = (3b-k)/2. If a1 matched, does a2 match better?
                    # No, for a fixed b and k, a1 and a2 give the same error k/2b.
                    # So we just need one match per b.
                    # We found the largest b for this k that has a match (via a1).
                    # We should also check if a2 gives a match at a larger b?
                    # The array `b_arr` is the same.
                    # If valid1 is false but valid2 is true, we need to check valid2.
                    
        if not found_any and np.any(valid2):
            a2_valid = a2[valid2]
            b_valid = b_arr[valid2]
            matches = sigma_np[a2_valid] == sigma_np[b_valid]
            if np.any(matches):
                idx = np.argmax(matches)
                b_best_k = int(b_valid[idx])
                a_best_k = int(a2_valid[idx])
                
                err = k / (2.0 * b_best_k)
                if err < best_err:
                    best_err = err
                    best_pair = (a_best_k, b_best_k)
        # End k loop
        
    # Verify result
    a_final, b_final = best_pair
    if not (1 <= a_final <= N and 1 <= b_final <= N):
        return [11, 6]
    if a_final == b_final:
        return [11, 6]
    if sigma_np[a_final] != sigma_np[b_final]:
        return [11, 6]
    
    err_final = abs(Fraction(a_final, b_final) - Fraction(3, 2))
    if err_final == 0:
        return [11, 6]
        
    return [int(a_final), int(b_final)]

# EVOLVE-BLOCK-END
