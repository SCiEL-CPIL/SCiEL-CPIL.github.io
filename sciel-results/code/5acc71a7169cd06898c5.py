import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    min_n = 1000
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # S[m] will store the small prime part of m
    # Size max_val + 1
    S = np.ones(max_val + 1, dtype=np.int64)
    
    # Sieve-like precomputation
    for p in primes:
        pk = p
        while pk <= max_val:
            # Multiply S[m] by p for all m divisible by pk
            # m = pk, 2*pk, 3*pk, ...
            # We can use numpy arange for speed
            multiples = np.arange(pk, max_val + 1, pk)
            S[multiples] *= p
            pk *= p
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
            
    if time.time() > deadline - 5:
        # Fallback: return a random or simple value if precomputation takes too long
        # But precomputation is fast.
        return min_n

    # Sliding window search
    # Window size 24
    # n ranges from min_n to max_n
    # Window for n is S[n+1] to S[n+24]
    
    # Initialize window for n = min_n
    # Indices: min_n+1 to min_n+24
    win_start_idx = min_n + 1
    win_end_idx = min_n + 24  # inclusive
    
    # freq dict: value -> count
    # Using a regular dict is fine for 1M steps
    # The values in S are integers.
    
    freq = {}
    distinct_count = 0
    
    # Add initial window elements
    for i in range(win_start_idx, win_end_idx + 1):
        val = int(S[i])
        if val in freq:
            freq[val] += 1
        else:
            freq[val] = 1
            distinct_count += 1
            
    best_count = distinct_count
    best_n = min_n
    
    # If best_count is 1, we can't do better, but let's verify if we should break.
    # The problem asks to MINIMIZE. 1 is the minimum possible.
    # So if we find 1, we can return immediately? 
    # Yes, 1 is the absolute minimum.
    if best_count == 1:
        return best_n

    # Slide the window
    # Next n is min_n + 1
    # Remove S[min_n+1], Add S[min_n+25]
    for n in range(min_n + 1, max_n + 1):
        # Remove element at index n (which is n+1 - 1 = n? No.)
        # Window for n-1 was S[n] to S[n+23].
        # Window for n is S[n+1] to S[n+24].
        # So we remove S[n] and add S[n+24].
        
        # Wait, my initialization:
        # n = min_n. Window S[min_n+1] ... S[min_n+24].
        # Next n = min_n + 1. Window S[min_n+2] ... S[min_n+25].
        # Removed: S[min_n+1]. Added: S[min_n+25].
        
        # In the loop variable n, we are at current n.
        # Previous n was n-1.
        # Window for n-1: S[n] ... S[n+23].
        # Window for n: S[n+1] ... S[n+24].
        # Remove S[n]. Add S[n+24].
        
        rem_val = int(S[n])
        add_val = int(S[n+24])
        
        # Remove rem_val
        cnt = freq[rem_val]
        if cnt == 1:
            del freq[rem_val]
            distinct_count -= 1
        else:
            freq[rem_val] = cnt - 1
            
        # Add add_val
        if add_val in freq:
            freq[add_val] += 1
        else:
            freq[add_val] = 1
            distinct_count += 1
            
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                return best_n
                
        if time.time() > deadline - 2:
            break
            
    return best_n
