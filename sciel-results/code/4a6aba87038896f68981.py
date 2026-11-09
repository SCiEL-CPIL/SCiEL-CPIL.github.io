# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    rank = m * p  # 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # V is Identity matrix of size rank x rank
    for r in range(rank):
        V[r][r] = 1
        
    # Fill U
    # U[i*m + j, r] = 1 if r in [j*p, j*p + p - 1]
    for i in range(n):
        for j in range(m):
            u_row = i * m + j
            start_col = j * p
            for k in range(p):
                U[u_row][start_col + k] = 1
                
    # Fill W
    # W[k*n + i, r] = 1 if r % p == k
    for k in range(p):
        for i in range(n):
            w_row = k * n + i
            for j in range(m):
                W[w_row][j * p + k] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
