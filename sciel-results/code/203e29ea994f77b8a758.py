DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Mixed decomposition: 2 blocks of Strassen (rank 7 each) + 1 block of naive (rank 4)
    # Total rank = 7 + 7 + 4 = 18
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen U vectors (shape 4)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_strassen = [
        [1, 0, 0, 1],   # P1: a00+a11
        [0, 0, 1, 1],   # P2: a10+a11
        [1, 0, 0, 0],   # P3: a00
        [0, 0, 0, 1],   # P4: a11
        [1, 1, 0, 0],   # P5: a00+a01
        [-1, 0, 1, 0],  # P6: a10-a00
        [0, 1, 0, -1]   # P7: a01-a11
    ]
    
    # Block 0: columns 0,1 of B -> Strassen (rank 7, cols 0-6)
    # Block 1: columns 2,3 of B -> Strassen (rank 7, cols 7-13)
    # Block 2: columns 4,5 of B -> Naive (rank 4, cols 14-17)
    
    for block in range(2):
        base_col = 2 * block  # 0 or 2
        strassen_offset = 7 * block  # 0 or 7
        
        for s in range(7):
            col = strassen_offset + s
            
            # Fill U
            for i in range(4):
                U[i][col] = U_strassen[s][i]
                
            # Fill V and W based on Strassen formulas
            # V indices: j*p + k. j in {0,1}, k in {base_col, base_col+1}
            # W indices: k*n + i. k in {base_col, base_col+1}, i in {0,1}
            
            if s == 0:    # P1: (a00+a11)(b00+b11)
                V[0 * p + base_col][col] = 1
                V[1 * p + base_col + 1][col] = 1
                W[base_col * n + 0][col] = 1
                W[(base_col + 1) * n + 1][col] = 1
            elif s == 1:  # P2: (a10+a11)b00
                V[0 * p + base_col][col] = 1
                W[base_col * n + 1][col] = 1
                W[(base_col + 1) * n + 1][col] = -1
            elif s == 2:  # P3: a00(b01-b11)
                V[0 * p + base_col + 1][col] = 1
                V[1 * p + base_col + 1][col] = -1
                W[(base_col + 1) * n + 0][col] = 1
                W[(base_col + 1) * n + 1][col] = 1
            elif s == 3:  # P4: a11(b10-b00)
                V[1 * p + base_col][col] = 1
                V[0 * p + base_col][col] = -1
                W[base_col * n + 0][col] = 1
                W[base_col * n + 1][col] = 1
            elif s == 4:  # P5: (a00+a01)b11
                V[1 * p + base_col + 1][col] = 1
                W[base_col * n + 0][col] = -1
                W[(base_col + 1) * n + 0][col] = 1
            elif s == 5:  # P6: (a10-a00)(b00+b01)
                V[0 * p + base_col][col] = 1
                V[0 * p + base_col + 1][col] = 1
                W[(base_col + 1) * n + 1][col] = 1
            elif s == 6:  # P7: (a01-a11)(b10+b11)
                V[1 * p + base_col][col] = 1
                V[1 * p + base_col + 1][col] = 1
                W[base_col * n + 0][col] = 1
    
    # Block 2: columns 4,5 of B -> Naive (rank 4, cols 14-17)
    base_col = 4
    naive_offset = 14
    
    # Naive: C[i][k] = sum_j A[i][j] B[j][k]
    # For each (i, j, k) combination, we need a rank-1 term
    # But we can group by (i, k) output position
    # C[0][4] = A[0][0]*B[0][4] + A[0][1]*B[1][4]
    # C[0][5] = A[0][0]*B[0][5] + A[0][1]*B[1][5]
    # C[1][4] = A[1][0]*B[0][4] + A[1][1]*B[1][4]
    # C[1][5] = A[1][0]*B[0][5] + A[1][1]*B[1][5]
    
    # Rank 4 naive: one term per (i, j) pair, but that's 4 terms total for 2x2x2
    # Actually for 2x2x2 naive, we have 4 multiplications:
    # M0 = A[0][0]*B[0][0], M1 = A[0][1]*B[1][0], M2 = A[1][0]*B[0][1], M3 = A[1][1]*B[1][1]
    # Wait, that's not right. Naive 2x2x2 has 4 scalar multiplications:
    # A[0][0]*B[0][0], A[0][1]*B[1][0], A[1][0]*B[0][1], A[1][1]*B[1][1]
    # No, that's also wrong. Let me think again.
    # C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0]
    # C[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1]
    # C[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0]
    # C[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1]
    # So we need 4 rank-1 terms, one for each (i,j) pair in A and corresponding (j,k) in B.
    # Term 0: A[0][0] * B[0][*] -> contributes to C[0][0] and C[0][1]
    # Term 1: A[0][1] * B[1][*] -> contributes to C[0][0] and C[0][1]
    # Term 2: A[1][0] * B[0][*] -> contributes to C[1][0] and C[1][1]
    # Term 3: A[1][1] * B[1][*] -> contributes to C[1][0] and C[1][1]
    
    # Term 0: col = naive_offset + 0
    col = naive_offset
    U[0][col] = 1  # A[0][0]
    V[0 * p + base_col][col] = 1    # B[0][4]
    V[0 * p + base_col + 1][col] = 1 # B[0][5]
    W[base_col * n + 0][col] = 1    # C[4][0]
    W[(base_col + 1) * n + 0][col] = 1 # C[5][0]
    
    # Term 1: col = naive_offset + 1
    col = naive_offset + 1
    U[1][col] = 1  # A[0][1]
    V[1 * p + base_col][col] = 1    # B[1][4]
    V[1 * p + base_col + 1][col] = 1 # B[1][5]
    W[base_col * n + 0][col] = 1    # C[4][0]
    W[(base_col + 1) * n + 0][col] = 1 # C[5][0]
    
    # Term 2: col = naive_offset + 2
    col = naive_offset + 2
    U[2][col] = 1  # A[1][0]
    V[0 * p + base_col][col] = 1    # B[0][4]
    V[0 * p + base_col + 1][col] = 1 # B[0][5]
    W[base_col * n + 1][col] = 1    # C[4][1]
    W[(base_col + 1) * n + 1][col] = 1 # C[5][1]
    
    # Term 3: col = naive_offset + 3
    col = naive_offset + 3
    U[3][col] = 1  # A[1][1]
    V[1 * p + base_col][col] = 1    # B[1][4]
    V[1 * p + base_col + 1][col] = 1 # B[1][5]
    W[base_col * n + 1][col] = 1    # C[4][1]
    W[(base_col + 1) * n + 1][col] = 1 # C[5][1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
