import time
import math

def construct():
    # Safety margin
    deadline = time.time() + 80
    target = 123457
    
    # 1. Compute Largest Prime Factor (LPF) for all numbers up to target
    max_n = target - 1
    lpf = [0] * (max_n + 1)
    lpf[1] = 1
    for i in range(2, max_n + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, max_n + 1, i):
                lpf[j] = i
                
    # 2. Find the pair (a, b) with a+b=target that minimizes max(P(a), P(b))
    best_max = float('inf')
    best_a = 1
    
    # We only need to check a up to target//2 due to symmetry
    # max(P(a), P(target-a)) is the same for (a, target-a) and (target-a, a)
    for a in range(1, target // 2 + 1):
        b = target - a
        cur_max = max(lpf[a], lpf[b])
        if cur_max < best_max:
            best_max = cur_max
            best_a = a
            
        # Optional: Early termination if we find P=2? (Unlikely for this N)
        # If best_max == 2, we can't do better (P(1)=1, but a,b > 0, so min P is 2 for even, 3 for odd? 
        # Actually P(1)=1. If a=1, P(a)=1. 
        # So min possible max is 1? No, b must be > 0.
        # If a=1, b=123456. P(b)=643.
        # If we find max=2, it means both a and b are powers of 2.
        # 2^a + 2^b = 123457 (odd). Impossible.
        # So min max is at least 3.
        
        # Check time
        if time.time() > deadline:
            break

    return [best_a, target - best_a]
