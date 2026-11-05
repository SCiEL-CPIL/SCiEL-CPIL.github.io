import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    target = 123457
    
    # Compute largest prime factor for all n up to target
    # P[n] will store the largest prime factor of n
    P = [0] * (target + 1)
    P[1] = 1
    
    # Sieve for largest prime factor
    # Handle 2 separately for efficiency
    for j in range(2, target + 1, 2):
        P[j] = 2
    
    # Iterate odd p from 3 to target
    for p in range(3, target + 1, 2):
        if P[p] == 0:  # p is prime
            for j in range(p, target + 1, p):
                P[j] = p
    
    # Now find a, b such that a+b=target and max(P[a], P[b]) is minimized
    # We can do this in O(target/2) by checking pairs (a, target-a) for a in [1, target//2]
    # and using symmetry
    min_max_p = float('inf')
    best_a = 1
    best_b = target - 1
    
    # Check a from 1 to target//2
    half_target = target // 2
    for a in range(1, half_target + 1):
        b = target - a
        pa = P[a]
        pb = P[b]
        current_max = pa if pa > pb else pb
        if current_max < min_max_p:
            min_max_p = current_max
            best_a = a
            best_b = b
            # Early termination: if min_max_p is very small, we can't do much better
            # The theoretical minimum is at least 2 (since target > 2)
            if min_max_p <= 2:
                break
            # Check time
            if time.time() > deadline:
                break
    
    return [best_a, best_b]
