"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin before 90s kill
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    best_n = 1000
    best_score = float('inf')
    
    max_val = 1000000 + 24
    smooth_parts = [1] * (max_val + 1)
    
    # Compute smooth part for each number efficiently
    for x in range(2, max_val + 1):
        temp = x
        sp = 1
        for p in primes:
            while temp % p == 0:
                sp *= p
                temp //= p
        smooth_parts[x] = sp
    
    # Sliding window approach
    # Window is n+1 to n+24 (24 elements)
    # Start with n=1000, window is 1001 to 1024
    
    window_counter = Counter()
    for i in range(1, 25):
        window_counter[smooth_parts[1000 + i]] += 1
    
    distinct_count = len(window_counter)
    if distinct_count < best_score:
        best_score = distinct_count
        best_n = 1000
    
    n = 1000
    while n <= 999999:  # n+24 <= 1000023, but we need n <= 1000000 so n+24 <= 1000024
        if time.time() > deadline:
            break
        
        # Slide window: remove n+1, add n+25
        n += 1
        old_val = smooth_parts[n]  # this was n+1 in previous window, now it's n (the one leaving)
        # Wait, let me re-think. Window for n is [n+1, n+24]
        # Window for n-1 is [n, n+23]
        # So going from n-1 to n: remove element at position n (which was n+1 for n-1... no)
        
        # Let me redo: for n, window is indices n+1, n+2, ..., n+24
        # For n-1, window is indices n, n+1, ..., n+23
        # Transition: remove index n, add index n+24
        
        # But I started at n=1000 with window 1001..1024
        # Next n=1001: window 1002..1025
        # Remove 1001, add 1025
        
        remove_val = smooth_parts[n]  # index n = 1001
        add_val = smooth_parts[n + 24]  # index n+24 = 1025
        
        window_counter[remove_val] -= 1
        if window_counter[remove_val] == 0:
            del window_counter[remove_val]
        
        window_counter[add_val] += 1
        
        distinct_count = len(window_counter)
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
