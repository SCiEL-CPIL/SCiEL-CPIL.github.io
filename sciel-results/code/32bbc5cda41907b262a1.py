import time
import math
import array

def construct():
    start_time = time.time()
    deadline = 55  # Safety margin before 60s limit
    
    limit = 1000000
    max_val = limit + 1
    
    # Use array module for memory efficiency and faster access
    # lpf[i] will store the largest prime factor of i
    # Initialize with 0, then fill in
    lpf = array.array('I', [0]) * (max_val + 1)
    
    # Sieve to find largest prime factor
    # For each prime p, update all multiples of p with p as a candidate
    # Since we go in increasing order, the last update gives the largest prime factor
    for p in range(2, max_val + 1):
        if lpf[p] == 0:
            # p is prime
            for j in range(p, max_val + 1, p):
                lpf[j] = p
    
    # Scan for the minimum max(LPF(n), LPF(n+1))
    best_n = 10000
    min_score = float('inf')
    
    for n in range(10000, limit + 1):
        score = max(lpf[n], lpf[n+1])
        if score < min_score:
            min_score = score
            best_n = n
    
    return best_n
