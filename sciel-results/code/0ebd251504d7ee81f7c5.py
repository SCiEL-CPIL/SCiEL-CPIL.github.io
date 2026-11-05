def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    # best_y_for_a[a] stores the minimum y > a such that inv[y] = a
    # If no such y exists, it remains INF
    INF = 10**9
    best_y_for_a = [INF] * (MOD + 1)
    
    for y in range(1, MOD):
        a = inv[y]
        if a < y:
            if y < best_y_for_a[a]:
                best_y_for_a[a] = y
                
    # Compute suffix minimums
    # suf[i] = min(best_y_for_a[i], best_y_for_a[i+1], ..., best_y_for_a[N])
    suf = [INF] * (MOD + 2)
    suf[MOD] = INF
    for i in range(MOD - 1, 0, -1):
        if best_y_for_a[i] < suf[i + 1]:
            suf[i] = best_y_for_a[i]
        else:
            suf[i] = suf[i + 1]
            
    best_start = 2
    best_len = 0
    
    for S in range(2, MOD): # S from 2 to 1007
        B = suf[S]
        if B == INF:
            E = 1007
        else:
            E = B - 1
            
        if E > 1007:
            E = 1007
            
        if E < S:
            continue
            
        current_len = E - S + 1
        if current_len > best_len:
            best_len = current_len
            best_start = S
            
    return [best_start, best_len]
