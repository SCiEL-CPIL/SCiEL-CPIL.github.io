import time

DIMENSIONS = [2, 2, 15]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for b in range(7):
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
            
        k1 = 2 * b
        k2 = 2 * b + 1
        
        # m1 = (a+d)(e+h)
        U[0][col] += 1
        U[3][col] += 1
        V[0 * p + k1][col] += 1
        V[1 * p + k2][col] += 1
        W[k1 * n + 0][col] += 1
        W[k2 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        W[k2 * n + 1][col] -= 1
        col += 1
        
        # m2 = (c+d)e
        U[2][col] += 1
        U[3][col] += 1
        V[0 * p + k1][col] += 1
        W[k1 * n + 1][col] += 1
        W[k2 * n + 1][col] -= 1
        col += 1
        
        # m3 = a(f-h)
        U[0][col] += 1
        V[0 * p + k2][col] += 1
        V[1 * p + k2][col] -= 1
        W[k2 * n + 0][col] += 1
        W[k2 * n + 1][col] += 1
        col += 1
        
        # m4 = d(g-e)
        U[3][col] += 1
        V[1 * p + k1][col] += 1
        V[0 * p + k1][col] -= 1
        W[k1 * n + 0][col] += 1
        W[k1 * n + 1][col] += 1
        col += 1
        
        # m5 = (a+b)h
        U[0][col] += 1
        U[1][col] += 1
        V[1 * p + k2][col] += 1
        W[k1 * n + 0][col] -= 1
        W[k2 * n + 0][col] += 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[2][col] += 1
        U[0][col] -= 1
        V[0 * p + k1][col] += 1
        V[0 * p + k2][col] += 1
        W[k2 * n + 1][col] += 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[1][col] += 1
        U[3][col] -= 1
        V[1 * p + k1][col] += 1
        V[1 * p + k2][col] += 1
        W[k1 * n + 0][col] += 1
        W[k2 * n + 0][col] -= 1
        col += 1
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    k1 = 14
    
    # p1 = a e
    U[0][col] += 1
    V[0 * p + k1][col] += 1
    W[k1 * n + 0][col] += 1
    col += 1
    
    # p2 = b g
    U[1][col] += 1
    V[1 * p + k1][col] += 1
    W[k1 * n + 0][col] += 1
    col += 1
    
    # p3 = c e
    U[2][col] += 1
    V[0 * p + k1][col] += 1
    W[k1 * n + 1][col] += 1
    col += 1
    
    # p4 = d g
    U[3][col] += 1
    V[1 * p + k1][col] += 1
    W[k1 * n + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
