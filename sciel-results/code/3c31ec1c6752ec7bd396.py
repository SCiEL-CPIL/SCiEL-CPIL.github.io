def construct():
    n, m, p = 2, 3, 10
    # 5 pairs of columns: (0,1), (2,3), (4,5), (6,7), (8,9)
    pairs = [(0,1), (2,3), (4,5), (6,7), (8,9)]
    
    rank = 30
    U = [[0]*rank for _ in range(n*m)]  # 6 x 30
    V = [[0]*rank for _ in range(m*p)]  # 30 x 30
    W = [[0]*rank for _ in range(p*n)]  # 20 x 30
    
    col = 0
    for j in range(m):
        for k1, k2 in pairs:
            for i in range(n):
                # u: nonzero at position i*m + j
                U[i*m + j][col] = 1
                # v: nonzero at positions j*p + k1 and j*p + k2
                V[j*p + k1][col] = 1
                V[j*p + k2][col] = 1
                # w: nonzero at positions k1*n + i and k2*n + i
                W[k1*n + i][col] = 1
                W[k2*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
