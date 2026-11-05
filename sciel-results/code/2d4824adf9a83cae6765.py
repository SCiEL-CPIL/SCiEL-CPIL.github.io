import time
import math

def construct():
    start_time = time.time()
    target = 123457
    
    # Fallback solution
    best_a, best_b = 1, target - 1
    
    # Compute largest prime factor for all n up to target
    P = [0] * (target + 1)
    P[1] = 1
    
    # Sieve for largest prime factor
    for p in range(2, target + 1):
        if P[p] == 0:
            for j in range(p, target + 1, p):
                P[j] = p
                
    min_max_p = float('inf')
    
    for a in range(1, target):
        b = target - a
        current_max = P[a] if P[a] > P[b] else P[b]
        if current_max < min_max_p:
            min_max_p = current_max
            best_a = a
            best_b = b
            
    return [best_a, best_b]
