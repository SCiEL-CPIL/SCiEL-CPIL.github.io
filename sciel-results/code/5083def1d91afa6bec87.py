import time
import math

def construct():
    deadline = time.time() + 75  # Safety margin before 90s limit
    target = 123457
    max_n = target - 1
    
    # Compute largest prime factor for all numbers up to target-1 using sieve
    lpf = [0] * (max_n + 1)
    lpf[1] = 1
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, max_n + 1, i):
                lpf[j] = i  # i is the largest prime factor since we go in increasing order
    
    best_max = float('inf')
    best_a = 1
    
    # Only need to check a up to target // 2 because (a, b) is equivalent to (b, a)
    for a in range(1, target // 2 + 1):
        if time.time() > deadline - 1:
            break
        b = target - a
        cur_max = max(lpf[a], lpf[b])
        if cur_max < best_max:
            best_max = cur_max
            best_a = a
    
    return [best_a, target - best_a]
