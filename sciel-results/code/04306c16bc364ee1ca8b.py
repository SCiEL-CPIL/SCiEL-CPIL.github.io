"""Matrix multiplication 2x3x8 with rank-minimal decomposition."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Rank 11 per 2x3x2 block (7 Strassen + 4 schoolbook for the extra column)
    # Total rank = 4 * 11 = 44
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # Indices for U: i*m + j
    # Indices for V: j*p + k
    # Indices for W: k*n + i
    
    block_rank = 11
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        base_col = block * block_rank
        
        # Strassen for A[0..1, 0..1] * B[0..1, k0..k1]
        # P1
        U[0*3 + 0][base_col + 0] += 1
        U[1*3 + 1][base_col + 0] += 1
        V[0*8 + k0][base_col + 0] += 1
        V[1*8 + k1][base_col + 0] += 1
        W[k0*2 + 0][base_col + 0] += 1
        W[k1*2 + 0][base_col + 0] += 1
        W[k0*2 + 1][base_col + 0] -= 1
        W[k1*2 + 1][base_col + 0] -= 1
        
        # P2
        U[1*3 + 0][base_col + 1] += 1
        U[1*3 + 1][base_col + 1] += 1
        V[0*8 + k0][base_col + 1] += 1
        W[k0*2 + 1][base_col + 1] += 1
        W[k1*2 + 1][base_col + 1] -= 1
        
        # P3
        U[0*3 + 0][base_col + 2] += 1
        V[1*8 + k0][base_col + 2] += 1
        V[1*8 + k1][base_col + 2] -= 1
        W[k1*2 + 0][base_col + 2] += 1
        W[k1*2 + 1][base_col + 2] += 1
        
        # P4
        U[1*3 + 1][base_col + 3] += 1
        V[0*8 + k1][base_col + 3] += 1
        V[0*8 + k0][base_col + 3] -= 1
        W[k0*2 + 0][base_col + 3] -= 1
        W[k1*2 + 0][base_col + 3] += 1
        W[k0*2 + 1][base_col + 3] -= 1
        W[k1*2 + 1][base_col + 3] += 1
        
        # P5
        U[0*3 + 0][base_col + 4] += 1
        U[0*3 + 1][base_col + 4] += 1
        V[1*8 + k1][base_col + 4] += 1
        W[k0*2 + 1][base_col + 4] += 1
        W[k1*2 + 1][base_col + 4] += 1
        
        # P6
        U[1*3 + 0][base_col + 5] += 1
        U[0*3 + 0][base_col + 5] -= 1
        V[0*8 + k0][base_col + 5] += 1
        V[0*8 + k1][base_col + 5] += 1
        W[k0*2 + 0][base_col + 5] += 1
        W[k1*2 + 0][base_col + 5] -= 1
        
        # P7
        U[0*3 + 1][base_col + 6] += 1
        U[1*3 + 1][base_col + 6] -= 1
        V[1*8 + k0][base_col + 6] += 1
        V[1*8 + k1][base_col + 6] += 1
        W[k0*2 + 0][base_col + 6] -= 1
        W[k1*2 + 0][base_col + 6] += 1
        W[k0*2 + 1][base_col + 6] -= 1
        W[k1*2 + 1][base_col + 6] += 1
        
        # P8: A[0,2] * B[2,k0] -> C[0,k0]
        U[0*3 + 2][base_col + 7] += 1
        V[2*8 + k0][base_col + 7] += 1
        W[k0*2 + 0][base_col + 7] += 1
        
        # P9: A[0,2] * B[2,k1] -> C[0,k1]
        U[0*3 + 2][base_col + 8] += 1
        V[2*8 + k1][base_col + 8] += 1
        W[k1*2 + 0][base_col + 8] += 1
        
        # P10: A[1,2] * B[2,k0] -> C[1,k0]
        U[1*3 + 2][base_col + 9] += 1
        V[2*8 + k0][base_col + 9] += 1
        W[k0*2 + 1][base_col + 9] += 1
        
        # P11: A[1,2] * B[2,k1] -> C[1,k1]
        U[1*3 + 2][base_col + 10] += 1
        V[2*8 + k1][base_col + 10] += 1
        W[k1*2 + 1][base_col + 10] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
