def construct():
    n, m, p = 2, 2, 13
    rank = m * p  # 26
    U = [[0] * rank for _ in range(n * m)]  # (4, 26)
    V = [[0] * rank for _ in range(m * p)]  # (26, 26)
    W = [[0] * rank for _ in range(p * n)]  # (26, 26)
    
    for k in range(p):
        for s in range(m):  # s in {0,1}
            r = k * m + s
            for j in range(m):
                for i in range(n):
                    # U[i*m + j, r] = delta(j, s)
                    if j == s:
                        U[i * m + j][r] = 1
                    # V[j*p + k, r] = delta(k, k) * delta(j, s) = delta(j, s)
                    if j == s:
                        V[j * p + k][r] = 1
                    # W[k*n + i, r] = 1 (for all i)
                    W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
