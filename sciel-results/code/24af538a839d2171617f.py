import math
import time

def construct():
    target = 123457
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin before 90s kill
    
    # Precompute largest prime factor for all numbers up to target
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (target + 1)
    
    # P(1) = 1
    if target >= 1:
        lpf[1] = 1
        
    # Sieve for largest prime factor
    # Iterate i from 2 to target
    # If lpf[i] is 0, i is prime
    # Then for all multiples j of i, set lpf[j] = i
    # Since i increases, the last assignment is the largest prime factor
    for i in range(2, target + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, target + 1, i):
                lpf[j] = i
    
    best_max_p = float('inf')
    best_a = 1
    best_b = target - 1
    
    # We can optimize the search slightly.
    # Since a+b is odd, one is even, one is odd.
    # The even number has factor 2.
    # The odd number (if >1) has odd prime factors.
    # We want to minimize the max prime factor.
    
    # Iterate a from 1 to target-1
    # We can stop at target // 2 + 1 because (a, b) and (b, a) yield the same max(P(a), P(b))
    half = target // 2 + 1
    for a in range(1, half):
        if time.time() > deadline:
            break
        b = target - a
        pa = lpf[a]
        pb = lpf[b]
        cur_max = pa if pa > pb else pb
        if cur_max < best_max_p:
            best_max_p = cur_max
            best_a = a
            best_b = b
            # Theoretical min for max(P(a),P(b)) given a+b is odd:
            # One is even (P>=2), one is odd (P>=3 if >1). So min max is 3.
            if best_max_p == 3:
                break
                
    return [best_a, best_b]
