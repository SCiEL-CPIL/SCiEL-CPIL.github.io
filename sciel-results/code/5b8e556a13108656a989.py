def construct():
    import time
    start = time.time()
    
    n, m, p = 2, 2, 9
    terms = []
    
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = k0 + 1
        
        U1 = {0: 1, 3: 1}
        V1 = {k0: 1, 9+k1: 1}
        W1 = {2*k0: 1, 2*k1+1: 1}
        terms.append((U1, V1, W1))
        
        U2 = {2: 1, 3: 1}
        V2 = {k0: 1}
        W2 = {2*k0+1: 1, 2*k1+1: -1}
        terms.append((U2, V2, W2))
        
        U3 = {0: 1}
        V3 = {k1: 1, 9+k1: -1}
        W3 = {2*k1: 1, 2*k1+1: 1}
        terms.append((U3, V3, W3))
        
        U4 = {3: 1}
        V4 = {9+k0: 1, k0: -1}
        W4 = {2*k0: 1, 2*k0+1: 1}
        terms.append((U4, V4, W4))
        
        U5 = {0: 1, 1: 1}
        V5 = {9+k1: 1}
        W5 = {2*k0: -1, 2*k1: 1}
        terms.append((U5, V5, W5))
        
        U6 = {2: 1, 0: -1}
        V6 = {k0: 1, k1: 1}
        W6 = {2*k1+1: 1}
        terms.append((U6, V6, W6))
        
        U7 = {1: 1, 3: -1}
        V7 = {9+k0: 1, 9+k1: 1}
        W7 = {2*k0: 1}
        terms.append((U7, V7, W7))
    
    k = 8
    terms.append(({0: 1}, {k: 1}, {2*k: 1}))
    terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
    terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
    terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    
    for t, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U[idx][t] = val
        for idx, val in Vd.items():
            V[idx][t] = val
        for idx, val in Wd.items():
            W[idx][t] = val
    
    return {"U": U, "V": V, "W": W}
