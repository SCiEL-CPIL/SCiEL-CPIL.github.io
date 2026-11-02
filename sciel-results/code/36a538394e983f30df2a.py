# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 6
    rank = m * p  # 18
    U = [[0] * rank for _ in range(n * m)]   # 6 x 18
    V = [[0] * rank for _ in range(m * p)]   # 18 x 18
    W = [[0] * rank for _ in range(p * n)]   # 12 x 18
    
    # For each k, we use 3 multiplications (r = 3k, 3k+1, 3k+2)
    # We use the identity decomposition for M_{2,3,1}
    # U_{i,j,r} = delta_{j, r%3}
    # V_{j,k,r} = delta_{j, r%3} if k == r//3 else 0
    # W_{k,i,r} = 1 if k == r//3 else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = k * m + j
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
