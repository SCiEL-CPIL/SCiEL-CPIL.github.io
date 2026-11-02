DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = []
    
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1
        u = [1, 0, 0, -1]
        v = [0] * (m * p)
        w = [0] * (p * n)
        v[1 * p + k0] = 1
        v[1 * p + k1] = 1
        w[k1 * n + 0] = 1
        terms.append((u, v, w))
        
        # M2
        u = [1, 0, 0, 1]
        v = [0] * (m * p)
        w = [0] * (p * n)
        v[1 * p + k1] = 1
        w[k0 * n + 0] = 1
        w[k1 * n + 1] = 1
        terms.append((u, v, w))
        
        # M3
        u = [1, 1, 0, 0]
        v = [0] * (m * p)
        w = [0] * (p * n)
        v[0 * p + k0] = 1
        w[k0 * n + 1] = 1
        terms.append((u, v, w))
        
        # M4
        u = [0, 0, 1, -1]
        v = [0] * (m * p)
        w = [0] * (p * n)
        v[0 * p + k0] = 1
        v[0 * p + k1] = 1
        w[k0 * n + 0] = 1
        w[k1 * n + 1] = -1
        terms.append((u, v, w))
        
        # M5
        u = [0, 1, 0, 1]
        v = [0] * (m * p)
        w = [0] * (p * n)
        v[1 * p + k0] = 1
        v[0 * p + k0] = -1
        w[k0 * n + 0] = 1
        terms.append((u, v, w))
        
        # M6
        u = [1, -1, 0, 0]
        v = [0] * (m * p)
        w = [0] * (p * n)
        v[0 * p + k0] = 1
        v[1 * p + k0] = 1
        w[k0 * n + 1] = 1
        terms.append((u, v, w))
        
        # M7
        u = [1, -1, 1, -1]
        v = [0] * (m * p)
        w = [0] * (p * n)
        v[0 * p + k0] = 1
        v[1 * p + k1] = 1
        w[k0 * n + 0] = -1
        w[k1 * n + 0] = 1
        terms.append((u, v, w))

    k0 = 14
    
    # P1 = A00 * B[0,0] -> C0
    u = [1, 0, 0, 0]
    v = [0] * (m * p)
    w = [0] * (p * n)
    v[0 * p + k0] = 1
    w[k0 * n + 0] = 1
    terms.append((u, v, w))
    
    # P2 = A01 * B[1,0] -> C0
    u = [0, 1, 0, 0]
    v = [0] * (m * p)
    w = [0] * (p * n)
    v[1 * p + k0] = 1
    w[k0 * n + 0] = 1
    terms.append((u, v, w))
    
    # P3 = A10 * B[0,0] -> C1
    u = [0, 0, 1, 0]
    v = [0] * (m * p)
    w = [0] * (p * n)
    v[0 * p + k0] = 1
    w[k0 * n + 1] = 1
    terms.append((u, v, w))
    
    # P4 = A11 * B[1,0] -> C1
    u = [0, 0, 0, 1]
    v = [0] * (m * p)
    w = [0] * (p * n)
    v[1 * p + k0] = 1
    w[k0 * n + 1] = 1
    terms.append((u, v, w))
    
    for r, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            U[i][r] = u[i]
        for j in range(m * p):
            V[j][r] = v[j]
        for k in range(p * n):
            W[k][r] = w[k]
            
    return {"U": U, "V": V, "W": W}
