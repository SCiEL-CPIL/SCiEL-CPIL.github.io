def construct():
    mod = 1009
    inv = [0] * 1009
    for x in range(1, 1009):
        inv[x] = pow(x, mod - 2, mod)
        
    INF = 10**9
    min_A_for_m = [INF] * 1009
    
    for x in range(1, 1009):
        m = min(x, inv[x])
        A = max(x, inv[x])
        if A < min_A_for_m[m]:
            min_A_for_m[m] = A
            
    M = [INF] * 1010
    M[1009] = INF
    for s in range(1008, 0, -1):
        M[s] = min(M[s + 1], min_A_for_m[s])
        
    best_start = 1
    best_len = 0
    
    for s in range(1, 1009):
        e_max = min(1008, M[s] - 1)
        if e_max >= s:
            length = e_max - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                
    return [best_start, best_len]
