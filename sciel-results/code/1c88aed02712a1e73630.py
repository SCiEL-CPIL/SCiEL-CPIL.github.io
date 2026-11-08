"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U1, V1, W1 are 4x7
    U1 = [
        [1, 0, 1, 0, 1, 0, 1], # a
        [0, 0, 0, 0, 1, 0, 0], # b
        [0, 1, 0, 0, 0, 1, -1], # c
        [0, 1, 0, 1, 0, -1, 0]  # d
    ]
    V1 = [
        [1, 1, 0, 0, 0, 0, 1], # e
        [0, 0, 1, -1, 0, 0, 0], # f
        [0, 0, 0, 1, 0, 1, 1], # g
        [1, 0, -1, 0, 1, 1, 0]  # h
    ]
    # W1 index: k*n + i. n=2.
    # 0: k=0, i=0 (C00)
    # 1: k=0, i=1 (C10) -- Wait. k is col, i is row.
    # C[i][k].
    # My previous mapping:
    # W row 0: C00.
    # W row 1: C10? No.
    # Let's re-verify W index mapping.
    # Task says: W[k*n+i, r].
    # k is column index of C. i is row index of C.
    # C is n x p. So i in 0..n-1, k in 0..p-1.
    # For 2x2x2, n=2, p=2.
    # i in 0..1, k in 0..1.
    # Index 0: k=0, i=0 -> C[0][0].
    # Index 1: k=0, i=1 -> C[1][0].
    # Index 2: k=1, i=0 -> C[0][1].
    # Index 3: k=1, i=1 -> C[1][1].
    
    # Strassen results:
    # C[0][0] = M1 + M4 - M5 + M7
    # C[1][0] = M2 + M4
    # C[0][1] = M3 + M5
    # C[1][1] = M1 - M2 + M3 + M6
    
    # So W1 rows:
    # Row 0 (C00): [1, 0, 0, 1, -1, 0, 1]
    # Row 1 (C10): [0, 1, 0, 1, 0, 0, 0]
    # Row 2 (C01): [0, 0, 1, 0, 1, 0, 0]
    # Row 3 (C11): [1, -1, 1, 0, 0, 1, 0]
    
    W1 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Linear map decomposition for 1x2x3
    # U2 is 2x6
    # V2 is 6x6
    # W2 is 3x6
    
    U2 = [
        [1, 0, 1, 0, 1, 0], # x0
        [0, 1, 0, 1, 0, 1]  # x1
    ]
    V2 = [
        [1, 0, 0, 0, 0, 0], # y00
        [0, 0, 1, 0, 0, 0], # y01
        [0, 0, 0, 0, 1, 0], # y02
        [0, 1, 0, 0, 0, 0], # y10
        [0, 0, 0, 1, 0, 0], # y11
        [0, 0, 0, 0, 0, 1]  # y12
    ]
    W2 = [
        [1, 1, 0, 0, 0, 0], # z0
        [0, 0, 1, 1, 0, 0], # z1
        [0, 0, 0, 0, 1, 1]  # z2
    ]
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Parameters
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 1, 2, 3
    
    # Indices
    # i in 0..1
    # j = j1*2 + j2, j1 in 0..1, j2 in 0..1
    # k = k1*3 + k2, k1 in 0..1, k2 in 0..2
    
    for i in range(n1):
        for j1 in range(m1):
            for j2 in range(m2):
                # U index
                row_U = i * m + j1 * m2 + j2
                u1_idx = i * m1 + j1
                u2_idx = j2
                
                for k1 in range(p1):
                    for k2 in range(p2):
                        # This loop is for W and V? No, U only depends on i, j.
                        # U is filled once per (i, j1, j2) and each c1, c2.
                        pass
                        
    # Fill U
    for i in range(n1):
        for j1 in range(m1):
            for j2 in range(m2):
                row_U = i * m + j1 * m2 + j2
                u1_idx = i * m1 + j1
                u2_idx = j2
                for c1 in range(7):
                    for c2 in range(6):
                        c = c1 * 6 + c2
                        U[row_U][c] = U1[u1_idx][c1] * U2[u2_idx][c2]
                        
    # Fill V
    for j1 in range(m1):
        for j2 in range(m2):
            for k1 in range(p1):
                for k2 in range(p2):
                    row_V = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                    v1_idx = j1 * p1 + k1
                    v2_idx = j2 * p2 + k2
                    for c1 in range(7):
                        for c2 in range(6):
                            c = c1 * 6 + c2
                            V[row_V][c] = V1[v1_idx][c1] * V2[v2_idx][c2]
                            
    # Fill W
    for k1 in range(p1):
        for k2 in range(p2):
            for i in range(n1):
                row_W = (k1 * p2 + k2) * n + i
                w1_idx = k1 * n1 + i
                w2_idx = k2
                for c1 in range(7):
                    for c2 in range(6):
                        c = c1 * 6 + c2
                        W[row_W][c] = W1[w1_idx][c1] * W2[w2_idx][c2]
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
