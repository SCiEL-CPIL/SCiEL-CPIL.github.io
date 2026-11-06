"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    MAX_N = 1000000
    WINDOW = 24
    MAX_VAL = MAX_N + WINDOW  # 1000024
    
    # Compute smooth part (product of prime factors < 24) for each number
    # Use a "remaining" approach: remaining[m] starts at m, we divide out small primes
    remaining = list(range(MAX_VAL + 1))
    smooth_part = [1] * (MAX_VAL + 1)
    
    for p in primes:
        for m in range(p, MAX_VAL + 1, p):
            cnt = 0
            r = remaining[m]
            while r % p == 0:
                r //= p
                cnt += 1
            if cnt > 0:
                remaining[m] = r
                smooth_part[m] *= (p ** cnt)
    
    # Sliding window to find min distinct smooth parts
    best_n = 1000
    best_count = float('inf')
    
    freq = defaultdict(int)
    
    # Initialize first window: n=1000, elements are smooth_part[1001..1024]
    for i in range(1001, 1001 + WINDOW):
        freq[smooth_part[i]] += 1
    
    distinct_count = len(freq)
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = 1000
    
    # Slide the window
    for n in range(1001, MAX_N + 1):
        remove_val = smooth_part[n]
        add_val = smooth_part[n + WINDOW - 1]
        
        freq[remove_val] -= 1
        if freq[remove_val] == 0:
            del freq[remove_val]
        
        freq[add_val] += 1
        
        distinct_count = len(freq)
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
        
        if n % 200000 == 0:
            if time.time() > deadline:
                break
    
    return best_n
# EVOLVE-BLOCK-END
