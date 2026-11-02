# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    # Base Strassen factors for 2x2x2
    # U0: 4x7, V0: 4x7, W0: 4x7
    # U0 rows: (0,0), (0,1), (1,0), (1,1)
    # V0 rows: (0,0), (0,1), (1,0), (1,1)
    # W0 rows: (0,0), (0,1), (1,0), (1,1) -- wait, W is p x n. For 2x2x2, p=2, n=2.
    # W index: k*n + i. k in {0,1}, i in {0,1}.
    # Rows: 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1).
    
    U0 = [[0]*7 for _ in range(4)]
    V0 = [[0]*7 for _ in range(4)]
    W0 = [[0]*7 for _ in range(4)]
    
    # Strassen factors
    # M1: (a00+a11)(b00+b11)
    # U: a00, a11 -> 1
    # V: b00, b11 -> 1
    # W: c00, c11 -> 1
    U0[0][0] = 1; U0[3][0] = 1
    V0[0][0] = 1; V0[3][0] = 1
    W0[0][0] = 1; W0[3][0] = 1
    
    # M2: (a10+a11)b00
    # U: a10, a11 -> 1
    # V: b00 -> 1
    # W: c10 -> 1, c11 -> -1
    U0[2][1] = 1; U0[3][1] = 1
    V0[0][1] = 1
    W0[2][1] = 1; W0[3][1] = -1
    
    # M3: a00(b01-b11)
    # U: a00 -> 1
    # V: b01 -> 1, b11 -> -1
    # W: c01 -> 1, c11 -> 1
    U0[0][2] = 1
    V0[1][2] = 1; V0[3][2] = -1
    W0[1][2] = 1; W0[3][2] = 1
    
    # M4: a11(b10-b00)
    # U: a11 -> 1
    # V: b10 -> 1, b00 -> -1
    # W: c00 -> 1, c10 -> 1
    U0[3][3] = 1
    V0[2][3] = 1; V0[0][3] = -1
    W0[0][3] = 1; W0[2][3] = 1
    
    # M5: (a00+a01)b11
    # U: a00, a01 -> 1
    # V: b11 -> 1
    # W: c00 -> -1, c01 -> 1
    U0[0][4] = 1; U0[1][4] = 1
    V0[3][4] = 1
    W0[0][4] = -1; W0[1][4] = 1
    
    # M6: (a10-a00)(b00+b01)
    # U: a10 -> 1, a00 -> -1
    # V: b00, b01 -> 1
    # W: c11 -> 1
    U0[2][5] = 1; U0[0][5] = -1
    V0[0][5] = 1; V0[1][5] = 1
    W0[3][5] = 1
    
    # M7: (a01-a11)(b10+b11)
    # U: a01 -> 1, a11 -> -1
    # V: b10, b11 -> 1
    # W: c00 -> 1
    U0[1][6] = 1; U0[3][6] = -1
    V0[2][6] = 1; V0[3][6] = 1
    W0[0][6] = 1

    # Construct full matrices
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            
            # U is same for all blocks
            for i in range(n):
                for j in range(m):
                    row_u = i * m + j
                    U[row_u][col] = U0[row_u][s]
            
            # V depends on block t
            for j in range(m):
                for k_local in range(2):
                    k_global = 2 * t + k_local
                    row_v = j * p + k_global
                    # V0 row index: j * 2 + k_local
                    row_v0 = j * 2 + k_local
                    V[row_v][col] = V0[row_v0][s]
            
            # W depends on block t
            for k_local in range(2):
                for i in range(n):
                    k_global = 2 * t + k_local
                    row_w = k_global * n + i
                    # W0 row index: k_local * 2 + i
                    row_w0 = k_local * 2 + i
                    W[row_w][col] = W0[row_w0][s]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
