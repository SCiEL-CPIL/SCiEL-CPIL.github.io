import time

def construct():
    N = 123457
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin

    # 1. Precompute largest prime factor for all numbers up to N
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    # Sieve for largest prime factor
    # i iterates through potential primes
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # 2. Search for the optimal split
    best_val = float('inf')
    best_a = 1
    
    for a in range(1, N):
        b = N - a
        # P(a) is lpf[a], P(b) is lpf[b]
        # Note: lpf[x] is the largest prime factor of x
        val = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        
        if val < best_val:
            best_val = val
            best_a = a
            
        # Check time
        if time.time() > deadline:
            break

    return [best_a, N - best_a]
