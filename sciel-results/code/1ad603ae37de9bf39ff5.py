def construct():
    n, m, p = 2, 3, 6
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12
    
    rank = 22
    U = np.zeros((dim_a, rank), dtype=np.int64)
    V = np.zeros((dim_b, rank), dtype=np.int64)
    W = np.zeros((dim_c, rank), dtype=np.int64)
    
    # Strassen 2x2x2 factors
    S_U = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    S_V = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    S_W = [
        [[1, 0], [0, 1]],
        [[0, 1], [0, -1]],
        [[0, 0], [1, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]
    
    col = 0
    # 3 blocks of Strassen for j=0,1 and k=2b,2b+1
    for b in range(3):
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = S_U[r][i][j]
            for j in range(2):
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[j * p + k_global, col] = S_V[r][j][kp]
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = S_W[r][kp][i]
            col += 1
    
    # Rank-1 term for j=2
    for i in range(2):
        U[i * m + 2, col] = 1
    for k in range(6):
        V[2 * p + k, col] = 1
    for k in range(6):
        for i in range(2):
            W[k * n + i, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
