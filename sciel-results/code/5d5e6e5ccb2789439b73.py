import time
import math

def construct():
    deadline = time.time() + 85
    
    TARGET = 123457
    
    # Sieve to compute largest prime factor for all numbers up to TARGET
    # lp[n] = largest prime factor of n, lp[1] = 1
    lp = [0] * (TARGET + 1)
    lp[1] = 1
    
    for i in range(2, TARGET + 1):
        if lp[i] == 0:  # i is prime
            for j in range(i, TARGET + 1, i):
                lp[j] = i  # i is a prime factor of j; since we go in increasing order, lp[j] ends up as the largest
    
    # Now scan all a from 1 to TARGET-1, compute b = TARGET - a
    # Objective: minimize max(lp[a], lp[b])
    best_val = float('inf')
    best_pair = (1, TARGET - 1)
    
    for a in range(1, TARGET):
        b = TARGET - a
        val = max(lp[a], lp[b])
        if val < best_val:
            best_val = val
            best_pair = (a, b)
            if best_val == 2:  # Can't do better than 2 (both powers of 2)
                break
    
    return list(best_pair)
