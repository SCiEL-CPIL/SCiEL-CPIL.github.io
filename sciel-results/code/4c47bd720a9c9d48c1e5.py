"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank = 55
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U indices:
    # 0: A[0,0]
    # 1: A[0,1]
    # 2: A[0,2]
    # 3: A[1,0]
    # 4: A[1,1]
    # 5: A[1,2]
    
    for j in range(5):
        r_base = 11 * j
        
        # Standard part (inner dim 2)
        # Term 0: A[0,2] * B[2,0] -> C[0,0]
        U[2][r_base + 0] = 1
        V[20 + 2*j][r_base + 0] = 1
        W[4*j][r_base + 0] = 1
        
        # Term 1: A[0,2] * B[2,1] -> C[0,1]
        U[2][r_base + 1] = 1
        V[21 + 2*j][r_base + 1] = 1
        W[4*j + 2][r_base + 1] = 1
        
        # Term 2: A[1,2] * B[2,0] -> C[1,0]
        U[5][r_base + 2] = 1
        V[20 + 2*j][r_base + 2] = 1
        W[4*j + 1][r_base + 2] = 1
        
        # Term 3: A[1,2] * B[2,1] -> C[1,1]
        U[5][r_base + 3] = 1
        V[21 + 2*j][r_base + 3] = 1
        W[4*j + 3][r_base + 3] = 1
        
        # Strassen part (inner dim 0,1)
        # V indices for block j:
        # B[0,0] -> 2*j
        # B[0,1] -> 2*j+1
        # B[1,0] -> 10+2*j
        # B[1,1] -> 11+2*j
        
        # Term 4: M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        U[0][r_base + 4] = 1
        U[4][r_base + 4] = 1
        V[2*j][r_base + 4] = 1
        V[11 + 2*j][r_base + 4] = 1
        W[4*j][r_base + 4] = 1
        W[4*j + 3][r_base + 4] = -1
        
        # Term 5: M2 = (A[1,0] + A[1,1]) * B[0,0]
        U[3][r_base + 5] = 1
        U[4][r_base + 5] = 1
        V[2*j][r_base + 5] = 1
        W[4*j + 1][r_base + 5] = 1
        W[4*j + 3][r_base + 5] = -1
        
        # Term 6: M3 = A[0,0] * (B[0,1] - B[1,1])
        U[0][r_base + 6] = 1
        V[2*j + 1][r_base + 6] = 1
        V[11 + 2*j][r_base + 6] = -1
        W[4*j + 2][r_base + 6] = 1
        W[4*j + 3][r_base + 6] = 1
        
        # Term 7: M4 = A[1,1] * (B[1,0] - B[0,0])
        U[4][r_base + 7] = 1
        V[10 + 2*j][r_base + 7] = 1
        V[2*j][r_base + 7] = -1
        W[4*j][r_base + 7] = 1
        W[4*j + 1][r_base + 7] = 1
        
        # Term 8: M5 = (A[0,0] + A[0,1]) * B[1,1]
        U[0][r_base + 8] = 1
        U[1][r_base + 8] = 1
        V[11 + 2*j][r_base + 8] = 1
        W[4*j][r_base + 8] = -1
        W[4*j + 2][r_base + 8] = 1
        
        # Term 9: M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        U[3][r_base + 9] = 1
        U[0][r_base + 9] = -1
        V[2*j][r_base + 9] = 1
        V[2*j + 1][r_base + 9] = 1
        W[4*j + 3][r_base + 9] = 1
        
        # Term 10: M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        U[1][r_base + 10] = 1
        U[4][r_base + 10] = -1
        V[10 + 2*j][r_base + 10] = 1
        V[11 + 2*j][r_base + 10] = 1
        W[4*j][r_base + 10] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
