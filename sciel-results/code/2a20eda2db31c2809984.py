import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    N = 123457
    
    # Sieve for Largest Prime Factor
    # lpf[i] will hold the largest prime factor of i
    lpf = [1] * (N + 1)
    
    # Optimization: Only iterate up to N
    for i in range(2, N + 1):
        if lpf[i] == 1: # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    # Scan all decompositions
    # Due to symmetry of max(P(a), P(N-a)), we only need to check a <= N/2
    # This reduces the search space by half.
    half_N = N // 2
    for a in range(1, half_N + 1):
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
        
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        
        # Calculate max prime factor for this pair
        m = pa if pa > pb else pb
        
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
            
    return [best_a, best_b]
