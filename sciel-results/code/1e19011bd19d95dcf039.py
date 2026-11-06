import math
import time

def construct():
    target = 123457
    start_time = time.time()
    deadline = start_time + 85

    # Precompute Largest Prime Factor (LPF) for all numbers up to target
    lpf = [0] * (target + 1)
    lpf[1] = 1
    
    for i in range(2, target + 1):
        if lpf[i] == 0:
            for j in range(i, target + 1, i):
                lpf[j] = i
                
    # Find the split a + b = target that minimizes max(lpf[a], lpf[b])
    best_max_p = float('inf')
    best_a = 1
    best_b = target - 1
    
    for a in range(1, target // 2 + 1):
        b = target - a
        pa = lpf[a]
        pb = lpf[b]
        cur_max = pa if pa > pb else pb
        if cur_max < best_max_p:
            best_max_p = cur_max
            best_a = a
            best_b = b
            
    return [best_a, best_b]
