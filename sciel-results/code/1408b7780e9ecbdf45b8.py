import math
import time

def construct():
    N = 123457
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # 1. Compute Largest Prime Factor (LPF) for all numbers up to N
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    # Sieve to find largest prime factor
    # For each prime i, mark its multiples with i as their largest prime factor
    # Since we iterate i in increasing order, the last prime i that divides j 
    # will be the largest prime factor.
    for i in range(2, N + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # 2. Prepare list of candidates a in [1, N-1]
    # We want to sort them by their LPF
    candidates = list(range(1, N))
    
    # Sort indices by lpf[index]
    # To handle ties in LPF, the order doesn't strictly matter for correctness 
    # of finding the *first* valid pair, as long as we check all with same LPF 
    # before moving to higher LPFs? 
    # Actually, my logic above: "When adding a, if b is in S, return".
    # If a and b have the same LPF, and we add a first, b is not in S. 
    # Then we add b, a is in S. We return. 
    # The max LPF is lpf[a] = lpf[b]. Correct.
    
    candidates.sort(key=lambda x: lpf[x])
    
    seen = set()
    
    # 3. Iterate and find the first pair
    for a in candidates:
        if time.time() > deadline:
             # Fallback to seed if timeout (unlikely)
             return [1, 123456]
             
        b = N - a
        if b in seen:
            return [a, b]
        seen.add(a)
        
    # Fallback (should not happen)
    return [1, 123456]
