def construct():
    start = time.time()
    deadline = start + 140.0

    n, m, p = DIMENSIONS  # 4, 4, 6
    
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for k in range(n):
        for j in range(m):
            for i in range(p):
                u_idx = k * m + j
                v_idx = j * p + i
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
