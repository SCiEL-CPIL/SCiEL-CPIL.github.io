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
    # Since N is small, this is very fast
    for a in range(1, N):
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
            
            # Theoretical minimum is 1, but 1+1=2 != N.
            # If we hit a very low value, we might still want to check for even lower?
            # No, we are minimizing. 1 is the absolute minimum for P(n).
            # If we find a pair with max(P)=1, it's impossible for N>2.
            # So we just keep scanning until we are sure or time runs out.
            # Given N=123457, full scan takes < 0.1s.
            
    return [best_a, best_b]
