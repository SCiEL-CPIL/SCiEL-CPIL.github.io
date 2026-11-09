def construct():
    import time
    n, m, p = 2, 3, 3
    deadline = time.time() + 100  # 100s deadline, 20s margin
    
    # Try to find rank-15 decomposition
    # ... [complex search code] ...
    
    # Fallback: rank-18 schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
