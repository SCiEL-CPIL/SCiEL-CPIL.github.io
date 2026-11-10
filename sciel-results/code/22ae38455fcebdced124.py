def construct():
    N = 123457
    lpf = [0] * (N + 1)
    # Sieve of Eratosthenes variant to find largest prime factor
    for i in range(2, N + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_a = 1
    best_score = float('inf')
    
    for a in range(1, N):
        b = N - a
        pa = lpf[a] if a > 1 else 1
        pb = lpf[b] if b > 1 else 1
        current_max = max(pa, pb)
        if current_max < best_score:
            best_score = current_max
            best_a = a
            
    return [best_a, N - best_a]
