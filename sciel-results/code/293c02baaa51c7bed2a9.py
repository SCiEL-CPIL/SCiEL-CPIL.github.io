"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    # We use a block decomposition:
    # A = [[A22, A21], [A12, A11]] where A22 is 2x2, A21 is 2x1, A12 is 1x2, A11 is 1x1
    # B = [[B28], [B18]] where B28 is 2x8, B18 is 1x8
    # C = A B
    # C_top = A22 B28 + A21 B18
    # C_bot = A12 B28 + A11 B18
    
    # Rank estimates:
    # A22 B28: 2x2x8. Use 4x Strassen (2x2x2). Rank = 4 * 7 = 28.
    # A21 B18: 2x1x8. Schoolbook. Rank = 2 * 1 * 8 = 16.
    # A12 B28: 1x2x8. Schoolbook. Rank = 1 * 2 * 8 = 16.
    # A11 B18: 1x1x8. Schoolbook. Rank = 1 * 1 * 8 = 8.
    # Total Rank = 28 + 16 + 16 + 8 = 68.
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A22 B28 using Strassen for 4 column pairs
    # A22 indices: i in [0,2), j in [0,2) -> U[i*3+j]
    # B28 indices: j in [0,2), k in [0,8) -> V[j*8+k]
    # C indices: i in [0,2), k in [0,8) -> W[k*3+i]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_U = [
        [1, 0, 1, 0], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [0, 0, 1, -1],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, -1, 0],# M3: B01-B11
        [0, 0, 1, -1],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    strassen_W = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, -1],# M2: C10, C11
        [0, 1, 0, 1], # M3: C01, C11
        [1, 0, 1, 0], # M4: C00, C10
        [-1, 1, 0, 0],# M5: C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    for block in range(4):
        k_start = block * 2
        for s in range(7):
            # U: 2x2 block of A
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = strassen_U[s][i * 2 + j]
            # V: 2x2 block of B (cols k_start, k_start+1)
            for j in range(2):
                for k in range(2):
                    V[j * p + k_start + k][col] = strassen_V[s][j * 2 + k]
            # W: 2x2 block of C (cols k_start, k_start+1)
            for i in range(2):
                for k in range(2):
                    W[(k_start + k) * n + i][col] = strassen_W[s][i * 2 + k]
            col += 1
            
    # Part 2: A21 B18 (Schoolbook)
    # A21: i in [0,2), j=2
    # B18: j=2, k in [0,8)
    # C: i in [0,2), k in [0,8)
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 3: A12 B28 (Schoolbook)
    # A12: i=2, j in [0,2)
    # B28: j in [0,2), k in [0,8)
    # C: i=2, k in [0,8)
    for j in range(2):
        for k in range(8):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    # Part 4: A11 B18 (Schoolbook)
    # A11: i=2, j=2
    # B18: j=2, k in [0,8)
    # C: i=2, k in [0,8)
    for k in range(8):
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
