def construct():
    n, m, p = DIMENSIONS
    rank = 76
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, j=0..3, k=0..7 (32 terms)
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: i=1,2 (44 terms)
    # j-split: [0], [1], [2,3]
    # k-split: [0,1], [2,3], [4,5], [6,7]
    
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # j-blocks: [0], [1], [2,3]
    j_blocks = [[0], [1], [2, 3]]
    k_blocks = [[0, 1], [2, 3], [4, 5], [6, 7]]
    
    for j_block in j_blocks:
        for k_block in k_blocks:
            if len(j_block) == 1 and len(k_block) == 2:
                # 1x1x2: 2 terms
                j = j_block[0]
                for k in k_block:
                    U[1 * m + j][col] = 1  # i=1
                    V[j * p + k][col] = 1
                    W[k * n + 1][col] = 1
                    col += 1
                    U[2 * m + j][col] = 1  # i=2
                    V[j * p + k][col] = 1
                    W[k * n + 2][col] = 1
                    col += 1
            elif len(j_block) == 2 and len(k_block) == 2:
                # 2x2x2: 7 Strassen terms
                j0, j1 = j_block
                k0, k1 = k_block
                for t in range(7):
                    for l in range(4):
                        if strassen_U[t][l] != 0:
                            li = l // 2
                            lj = l % 2
                            gj = j0 if lj == 0 else j1
                            U[(1 + li) * m + gj][col] += strassen_U[t][l]
                        if strassen_V[t][l] != 0:
                            lj = l // 2
                            lk = l % 2
                            gj = j0 if lj == 0 else j1
                            gk = k0 if lk == 0 else k1
                            V[gj * p + gk][col] += strassen_V[t][l]
                        if strassen_W[t][l] != 0:
                            li = l // 2
                            lk = l % 2
                            gk = k0 if lk == 0 else k1
                            W[gk * n + (1 + li)][col] += strassen_W[t][l]
                    col += 1
    
    return {"U": U, "V": V, "W": W}
