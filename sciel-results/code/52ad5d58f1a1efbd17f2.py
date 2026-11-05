def construct():
    n, m, p = 4, 4, 7
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 16 x 28
    V = [[0] * rank for _ in range(m * p)]  # 28 x 28
    W = [[0] * rank for _ in range(p * n)]  # 28 x 28
    
    col = 0
    for j in range(m):  # j = 0..3
        for k in range(p):  # k = 0..6
            r = col
            col += 1
            # U[i*m+j, r] = 1 for all i (independent of i)
            for i in range(n):
                U[i * m + j][r] = 1
            # V[j'*p+k, r] = 1 if j'=j, else 0
            V[j * p + k][r] = 1
            # W[k'*n+i', r] = 1 if k'=k, else 0 (independent of i')
            for i_prime in range(n):
                W[k * n + i_prime][r] = 1
    
    return {"U": U, "V": V, "W": W}
