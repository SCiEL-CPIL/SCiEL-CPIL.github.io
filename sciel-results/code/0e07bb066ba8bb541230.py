import time
import sys
from collections import defaultdict

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N = 1000000
    MAX_VAL = N + 24
    
    # Precompute the small part for all numbers up to MAX_VAL
    # small_part[x] will be the factor of x composed of primes in 'primes'
    sp = [1] * (MAX_VAL + 1)
    
    for p in primes:
        power = p
        while power <= MAX_VAL:
            for m in range(power, MAX_VAL + 1, power):
                sp[m] *= p
            power *= p
    
    # Now sp[i] contains the complete factor of i composed of primes strictly below 24.
    
    best_n = 1000
    min_distinct = 100  # Start high
    start_time = time.time()
    deadline = 85  # Leave 5s margin for overhead/return
    
    # Sliding window approach with frequency count
    # Window is sp[n+1], sp[n+2], ..., sp[n+24] for n in [1000, 1000000]
    # So we need sp[1001] to sp[1000024]
    
    # Initialize frequency dict for the first window: sp[1001] to sp[1024]
    freq = defaultdict(int)
    distinct_count = 0
    
    # Build initial window
    for i in range(1001, 1025):
        val = sp[i]
        if freq[val] == 0:
            distinct_count += 1
        freq[val] += 1
    
    min_distinct = distinct_count
    best_n = 1000
    
    if min_distinct == 1:
        return best_n
    
    # Slide the window
    # For n from 1001 to N:
    #   Remove sp[n], add sp[n+24]
    #   Window is sp[n+1] to sp[n+24]
    for n in range(1001, N + 1):
        if (n & 4095) == 0:
            if time.time() - start_time > deadline:
                break
        
        # Remove sp[n]
        val_out = sp[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            distinct_count -= 1
        
        # Add sp[n+24]
        val_in = sp[n + 24]
        if freq[val_in] == 0:
            distinct_count += 1
        freq[val_in] += 1
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = n
            
            if min_distinct == 1:
                break
    
    return best_n

construct()
