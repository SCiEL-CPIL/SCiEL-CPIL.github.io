import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    min_n = 1000
    max_n = 1000000
    max_val = max_n + 24
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    S = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        pk = p
        while pk <= max_val:
            multiples = np.arange(pk, max_val + 1, pk)
            S[multiples] *= p
            pk *= p
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    if time.time() > deadline - 5:
        return min_n
    
    # Map S values to compact indices for faster frequency tracking
    # Get unique values and their indices
    unique_vals, inverse = np.unique(S, return_inverse=True)
    num_unique = len(unique_vals)
    
    # freq array indexed by compact index
    freq = np.zeros(num_unique, dtype=np.int32)
    
    # Initial window: S[min_n+1] to S[min_n+24]
    init_idx = inverse[min_n + 1: min_n + 25]
    for idx in init_idx:
        freq[idx] += 1
    distinct_count = np.count_nonzero(freq)
    
    best_count = distinct_count
    best_n = min_n
    
    if best_count == 1:
        return best_n
    
    # Slide window
    # For n from min_n+1 to max_n:
    # Remove S[n], add S[n+24]
    # inverse[n] is the compact index for S[n]
    # inverse[n+24] is the compact index for S[n+24]
    
    inv = inverse  # alias for speed
    
    for n in range(min_n + 1, max_n + 1):
        rem_idx = inv[n]
        add_idx = inv[n + 24]
        
        if rem_idx == add_idx:
            # Same value, no change in distinct count
            pass
        else:
            freq[rem_idx] -= 1
            if freq[rem_idx] == 0:
                distinct_count -= 1
            freq[add_idx] += 1
            if freq[add_idx] == 1:
                distinct_count += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                return best_n
        
        if n % 100000 == 0 and time.time() > deadline - 2:
            break
    
    return best_n
