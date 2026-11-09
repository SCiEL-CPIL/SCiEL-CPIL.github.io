def construct():
    import time
    
    N = 123457
    start_time = time.time()
    DEADLINE = 85  # Safety margin
    
    # Sieve to find the largest prime factor for each number up to N
    # lpf[i] will store the largest prime factor of i
    lpf = [1] * (N + 1)
    
    # Standard sieve modification for largest prime factor
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    # Scan all possible decompositions a + b = N
    for a in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        
        # Calculate max prime factor for this pair
        current_max = pa if pa > pb else pb
        
        if current_max < best_max_pf:
            best_max_pf = current_max
            best_a = a
            best_b = b
            
            # Early exit if we found the theoretical minimum for non-trivial cases?
            # P(1)=1. If a=1, b=N. P(N) might be large.
            # If we find a pair with max prime factor 2, it's likely optimal unless 1 is involved.
            # But 1 is a valid positive integer. P(1)=1.
            # If a=1, max(1, P(N)).
            # If we find max prime factor = 2, it's better than any prime > 2.
            # Can we have max prime factor = 1? Only if a=1 and b=1, sum=2 != N.
            # So min possible is 2.
            if best_max_pf == 2:
                break

    return [best_a, best_b]
