import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # 16 for 2x1x8 part
    # 28 for 2x2x8 part (4 * 7 Strassen)
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x1x8 multiplication
    # A1 is 2x1, B1 is 1x8
    # A1[i, 0] * B1[0, k] -> C1[i, k]
    # Global indices:
    # U: i in 0..1, j=0 -> row i
    # V: j=0, k in 0..7 -> row k
    # W: k in 0..7, i in 0..1 -> row 2k + i
    
    for i in range(n):
        for k in range(p):
            U[i][col] = 1
            V[k][col] = 1
            W[2 * k + i][col] = 1
            col += 1
            
    # Part 2: 2x2x8 multiplication
    # A2 is 2x2, B2 is 2x8
    # Split B2 into 4 blocks of 2x2
    # B2[:, 4q:4q+2]
    # A2[i, j] * B2[j, k] -> C2[i, k]
    # Global indices:
    # U: i in 0..1, j in 1..2 -> row 2i + j
    # V: j in 1..2, k in 4q..4q+1 -> row 8j + k
    # W: k in 4q..4q+1, i in 0..1 -> row 2k + i
    
    for q in range(4):
        k_base = 4 * q
        
        # M1 = (A[0,1] + A[1,2]) * (B[1, k0] + B[2, k1])
        # U: A[0,1] (row 1), A[1,2] (row 3)
        # V: B[1, k0] (row 8+k0), B[2, k1] (row 16+k1)
        # W: C[0, k0] (2k0), C[1, k0] (2k0+1), C[0, k1] (2k1), C[1, k1] (2k1+1)
        # Coeffs: 1, 1, -1, -1
        U[1][col] = 1
        U[3][col] = 1
        V[8 + k_base][col] = 1
        V[16 + k_base + 1][col] = 1
        W[2 * k_base][col] = 1
        W[2 * k_base + 1][col] = 1
        W[2 * (k_base + 1)][col] = -1
        W[2 * (k_base + 1) + 1][col] = -1
        col += 1
        
        # M2 = (A[1,1] + A[1,2]) * B[1, k0]
        # U: A[1,1] (row 2), A[1,2] (row 3)
        # V: B[1, k0] (row 8+k0)
        # W: C[1, k0] (2k0+1), C[1, k1] (2k1+1)
        # Coeffs: 1, 1
        U[2][col] = 1
        U[3][col] = 1
        V[8 + k_base][col] = 1
        W[2 * k_base + 1][col] = 1
        W[2 * (k_base + 1) + 1][col] = 1
        col += 1
        
        # M3 = A[0,1] * (B[1, k1] - B[2, k1])
        # U: A[0,1] (row 1)
        # V: B[1, k1] (row 8+k1), B[2, k1] (row 16+k1)
        # W: C[0, k1] (2k1), C[1, k1] (2k1+1)
        # Coeffs: 1, 1
        U[1][col] = 1
        V[8 + k_base + 1][col] = 1
        V[16 + k_base + 1][col] = -1
        W[2 * (k_base + 1)][col] = 1
        W[2 * (k_base + 1) + 1][col] = 1
        col += 1
        
        # M4 = A[1,2] * (B[2, k0] - B[1, k0])
        # U: A[1,2] (row 3)
        # V: B[2, k0] (row 16+k0), B[1, k0] (row 8+k0)
        # W: C[0, k0] (2k0), C[1, k0] (2k0+1)
        # Coeffs: 1, 1
        U[3][col] = 1
        V[16 + k_base][col] = 1
        V[8 + k_base][col] = -1
        W[2 * k_base][col] = 1
        W[2 * k_base + 1][col] = 1
        col += 1
        
        # M5 = (A[0,1] + A[0,2]) * B[2, k1]
        # U: A[0,1] (row 1), A[0,2] (row 2)
        # V: B[2, k1] (row 16+k1)
        # W: C[0, k0] (2k0), C[0, k1] (2k1)
        # Coeffs: -1, 1
        U[1][col] = 1
        U[2][col] = 1
        V[16 + k_base + 1][col] = 1
        W[2 * k_base][col] = -1
        W[2 * (k_base + 1)][col] = 1
        col += 1
        
        # M6 = (A[1,1] - A[0,1]) * (B[1, k0] + B[1, k1])
        # U: A[1,1] (row 2), A[0,1] (row 1)
        # V: B[1, k0] (row 8+k0), B[1, k1] (row 8+k1)
        # W: C[1, k1] (2k1+1)
        # Coeffs: 1
        U[2][col] = 1
        U[1][col] = -1
        V[8 + k_base][col] = 1
        V[8 + k_base + 1][col] = 1
        W[2 * (k_base + 1) + 1][col] = 1
        col += 1
        
        # M7 = (A[0,2] - A[1,2]) * (B[2, k0] + B[2, k1])
        # U: A[0,2] (row 2), A[1,2] (row 3)
        # V: B[2, k0] (row 16+k0), B[2, k1] (row 16+k1)
        # W: C[0, k0] (2k0)
        # Coeffs: 1
        U[2][col] = 1
        U[3][col] = -1
        V[16 + k_base][col] = 1
        V[16 + k_base + 1][col] = 1
        W[2 * k_base][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
