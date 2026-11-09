def construct():
    import time
    start_time = time.time()
    # Safety margin: Evaluator kills at 90s. We use 85s.
    DEADLINE = 85
    
    N = 123457
    
    # Sieve for largest prime factor
    # lpf[i] will store the largest prime factor of i
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    # Scan all decompositions a + b = N
    # We can scan up to N//2 due to symmetry
    for a in range(1, N // 2 + 1):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
            # Theoretical lower bound: since N is odd, one of a,b is even (P>=2) 
            # and the other is odd (P>=3 if >1). So min possible is 3.
            if best_max_pf <= 3:
                break
    
    return [best_a, best_b]
