def construct():
    n, m, p = 2, 5, 6
    rank = n * m * (p // 2) * 2  # = 2 * 5 * 3 * 2 = 60? No...
    
    # Actually: for each j (5 values), for each k_block (3 blocks of size 2), rank 2
    # Total = 5 * 3 * 2 = 30
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k_block in range(3):
            k0 = k_block * 2
            k1 = k_block * 2 + 1
            # Term 1: u=[1,0], w=[1,1], v[j]=1
            U[0*m + j][col] += 1
            V[j*p + k0][col] += 1
            V[j*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 0][col] += 1
            col += 1
            # Term 2: u=[0,1], w=[1,1], v[j]=1
            U[1*m + j][col] += 1
            V[j*p + k0][col] += 1
            V[j*p + k1][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
