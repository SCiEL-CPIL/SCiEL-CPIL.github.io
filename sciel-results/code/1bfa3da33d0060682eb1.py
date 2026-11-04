DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    
    # Strassen 2x2x2 decomposition
    # 7 terms for a 2x2x2 block
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1 - M2 - M4 + M5
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M3 + M6 - M7
    
    # U coefficients (for a_i_j): 2x2 matrix for each term
    strassen_U = [
        [1, 0],  # M1: a00+a11
        [0, 1],  # M2: a10+a11 -> wait, this is wrong
    ]
    
    # Let me redo this. U[i*m+j, c] where i,j are local 2x2 indices
    # M1: (a00+a11) -> U[0,0]=1, U[1,1]=1, others 0
    # M2: (a10+a11) -> U[1,0]=1, U[1,1]=1, others 0
    # M3: a00 -> U[0,0]=1, others 0
    # M4: a11 -> U[1,1]=1, others 0
    # M5: (a00+a01) -> U[0,0]=1, U[0,1]=1, others 0
    # M6: (a10-a00) -> U[1,0]=1, U[0,0]=-1, others 0
    # M7: (a01-a11) -> U[0,1]=1, U[1,1]=-1, others 0
    
    # V coefficients (for b_j_k): 2x2 matrix for each term
    # M1: (b00+b11) -> V[0,0]=1, V[1,1]=1
    # M2: b00 -> V[0,0]=1
    # M3: (b01-b11) -> V[0,1]=1, V[1,1]=-1
    # M4: (b10-b00) -> V[1,0]=1, V[0,0]=-1
    # M5: b11 -> V[1,1]=1
    # M6: (b00+b01) -> V[0,0]=1, V[0,1]=1
    # M7: (b10+b11) -> V[1,0]=1, V[1,1]=1
    
    # W coefficients (for c_i_k): 2x2 matrix for each term
    # c00 = M1 - M2 - M4 + M5
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M3 + M6 - M7
    # M1: W[0,0]=1, W[1,1]=1
    # M2: W[0,0]=-1, W[1,0]=1
    # M3: W[0,1]=1, W[1,1]=-1
    # M4: W[0,0]=-1, W[1,0]=1
    # M5: W[0,0]=1, W[0,1]=1
    # M6: W[1,1]=1
    # M7: W[1,1]=-1
    
    strassen_U = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, 1]],  # M2
        [[1, 0], [0, 0]],  # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]], # M7
    ]
    
    strassen_V = [
        [[1, 0], [0, 1]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 1], [-1, 0]], # M3
        [[-1, 0], [1, 0]], # M4
        [[0, 0], [0, 1]],  # M5
        [[1, 1], [0, 0]],  # M6
        [[0, 0], [1, 1]],  # M7
    ]
    
    strassen_W = [
        [[1, 0], [0, 1]],  # M1
        [[-1, 0], [1, 0]], # M2
        [[0, 1], [-1, 0]], # M3
        [[-1, 0], [1, 0]], # M4
        [[1, 1], [0, 0]],  # M5
        [[0, 0], [0, 1]],  # M6
        [[0, 0], [0, -1]], # M7
    ]
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 6 x 39
    V = [[0] * rank for _ in range(m * p)]  # 21 x 39
    W = [[0] * rank for _ in range(p * n)]  # 14 x 39
    
    col = 0
    
    # Three Strassen blocks: (j=0,1; k=0,1), (j=0,1; k=2,3), (j=0,1; k=4,5)
    for block in range(3):
        k_offset = block * 2  # 0, 2, 4
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = strassen_U[c][i][j]
            for j in range(2):
                for k in range(2):
                    V[j * p + (k_offset + k)][col] = strassen_V[c][j][k]
            for k in range(2):
                for i in range(2):
                    W[(k_offset + k) * n + i][col] = strassen_W[c][i][k]
            col += 1
    
    # Schoolbook for j=2, k=0..6
    for i in range(2):
        for k in range(7):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Schoolbook for j=0,1; k=6
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
