def construct():
    N = 123457
    # 1. Compute Largest Prime Factor for all i in [1, N]
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # 2. Find min max(P(a), P(N-a))
    best_score = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        if current_max < best_score:
            best_score = current_max
            best_a = a
            best_b = b
            
    return [best_a, best_b]
