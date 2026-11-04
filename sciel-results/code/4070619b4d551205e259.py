DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # 2x2x3 rank-4 decomposition
    # U_small: (4, 4) - rows (i,a) for i in {0,1}, a in {0,1}
    U_small = [
        [1, 0, 0, 0],  # (i=0,a=0)
        [0, 1, 0, 0],  # (i=0,a=1)
        [0, 0, 1, 0],  # (i=1,a=0)
        [0, 0, 0, 1],  # (i=1,a=1)
    ]
    
    # V_small: (6, 4) - rows (a,k) for a in {0,1}, k in {0,1,2}
    V_small = [
        [1, 0, 1, 0],  # (a=0,k=0)
        [1, 0, 1, 0],  # (a=0,k=1)
        [1, 0, 1, 0],  # (a=0,k=2)
        [0, 1, 0, 1],  # (a=1,k=0)
        [0, 1, 0, 1],  # (a=1,k=1)
        [0, 1, 0, 1],  # (a=1,k=2)
    ]
    
    # W_small: (6, 4) - rows (i,k) for i in {0,1}, k in {0,1,2}
    W_small = [
        [1, 1, 0, 0],  # (i=0,k=0)
        [1, 1, 0, 0],  # (i=0,k=1)
        [1, 1, 0, 0],  # (i=0,k=2)
        [0, 0, 1, 1],  # (i=1,k=0)
        [0, 0, 1, 1],  # (i=1,k=1)
        [0, 0, 1, 1],  # (i=1,k=2)
    ]
    
    rank_small = 4
    rank = rank_small * rank_small  # 16
    
    # Build U_big: (16, 16)
    U = [[0] * rank for _ in range(n * m)]  # 16 x 16
    
    # Build V_big: (36, 16)
    V = [[0] * rank for _ in range(m * p)]  # 36 x 16
    
    # Build W_big: (36, 16)
    W = [[0] * rank for _ in range(p * n)]  # 36 x 16
    
    # U_big[i, a, (c1, c2)] = U_small[i1, a1, c1] * U_small[i2, a2, c2]
    # where i = 2*i1 + i2, a = 2*a1 + a2
    for i1 in range(2):
        for i2 in range(2):
            i = 2 * i1 + i2
            for a1 in range(2):
                for a2 in range(2):
                    a = 2 * a1 + a2
                    row_idx = i * m + a  # i * 4 + a
                    for c1 in range(rank_small):
                        for c2 in range(rank_small):
                            col_idx = c1 * rank_small + c2
                            U[row_idx][col_idx] = U_small[i1 * 2 + a1][c1] * U_small[i2 * 2 + a2][c2]
    
    # V_big[a, k, (c1, c2)] = V_small[a1, k1, c1] * V_small[a2, k2, c2]
    # where a = 2*a1 + a2, k = 3*k1 + k2
    for a1 in range(2):
        for a2 in range(2):
            a = 2 * a1 + a2
            for k1 in range(3):
                for k2 in range(3):
                    k = 3 * k1 + k2
                    row_idx = a * p + k  # a * 9 + k
                    for c1 in range(rank_small):
                        for c2 in range(rank_small):
                            col_idx = c1 * rank_small + c2
                            V[row_idx][col_idx] = V_small[a1 * 3 + k1][c1] * V_small[a2 * 3 + k2][c2]
    
    # W_big[i, k, (c1, c2)] = W_small[i1, k1, c1] * W_small[i2, k2, c2]
    # where i = 2*i1 + i2, k = 3*k1 + k2
    for i1 in range(2):
        for i2 in range(2):
            i = 2 * i1 + i2
            for k1 in range(3):
                for k2 in range(3):
                    k = 3 * k1 + k2
                    row_idx = k * n + i  # k * 4 + i
                    for c1 in range(rank_small):
                        for c2 in range(rank_small):
                            col_idx = c1 * rank_small + c2
                            W[row_idx][col_idx] = W_small[i1 * 3 + k1][c1] * W_small[i2 * 3 + k2][c2]
    
    return {"U": U, "V": V, "W": W}
