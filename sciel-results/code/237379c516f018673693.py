"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-44 exact tensor decomposition for 2x3x8 matrix multiplication.
    Decomposes into 4 blocks of 2x3x2, each using Strassen (rank 7) + 4 rank-1 terms.
    """
    n, m, p = DIMENSIONS
    rank = 44  # 4 blocks * 11 terms per block
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # For each 2-column block of the 8 columns
    for b in range(4):
        k0 = 2 * b      # first column index in this block
        k1 = 2 * b + 1  # second column index in this block
        offset = 11 * b # rank offset for this block
        
        # Strassen terms for 2x2 submatrix (rows 0,1 of U; cols k0,k1 of V)
        # U indices: row i (0,1), col j (0,1) -> U[i*m + j]
        # V indices: row j (0,1,2), col k (k0,k1) -> V[j*p + k]
        # W indices: row k (k0,k1), col i (0,1) -> W[k*n + i]
        
        # Term 1: M1 = (a00 + a11) * (b0k0 + b1k1)
        # c00 += M1, c11 += M1
        # U: a00, a11
        U[0*m + 0][offset + 0] = 1
        U[1*m + 1][offset + 0] = 1
        # V: b0k0, b1k1
        V[0*p + k0][offset + 0] = 1
        V[1*p + k1][offset + 0] = 1
        # W: c00, c11
        W[k0*n + 0][offset + 0] = 1
        W[k1*n + 1][offset + 0] = 1
        
        # Term 2: M2 = (a10 + a11) * b0k0
        # c10 += M2
        U[1*m + 0][offset + 1] = 1
        U[1*m + 1][offset + 1] = 1
        V[0*p + k0][offset + 1] = 1
        W[k1*n + 0][offset + 1] = 1
        
        # Term 3: M3 = a00 * (b0k1 - b1k1)
        # c01 += M3, c11 -= M3
        U[0*m + 0][offset + 2] = 1
        V[0*p + k1][offset + 2] = 1
        V[1*p + k1][offset + 2] = -1
        W[k0*n + 1][offset + 2] = 1
        W[k1*n + 1][offset + 2] = -1
        
        # Term 4: M4 = a11 * (b1k0 - b0k0)
        # c00 -= M4, c10 += M4
        U[1*m + 1][offset + 3] = 1
        V[1*p + k0][offset + 3] = 1
        V[0*p + k0][offset + 3] = -1
        W[k0*n + 0][offset + 3] = -1
        W[k1*n + 0][offset + 3] = 1
        
        # Term 5: M5 = (a00 + a01) * b1k1
        # c01 += M5
        U[0*m + 0][offset + 4] = 1
        U[0*m + 1][offset + 4] = 1
        V[1*p + k1][offset + 4] = 1
        W[k0*n + 1][offset + 4] = 1
        
        # Term 6: M6 = (a10 - a00) * (b0k0 + b0k1)
        # c00 += M6
        U[1*m + 0][offset + 5] = 1
        U[0*m + 0][offset + 5] = -1
        V[0*p + k0][offset + 5] = 1
        V[0*p + k1][offset + 5] = 1
        W[k0*n + 0][offset + 5] = 1
        
        # Term 7: M7 = (a01 - a11) * (b1k0 + b1k1)
        # c11 += M7
        U[0*m + 1][offset + 6] = 1
        U[1*m + 1][offset + 6] = -1
        V[1*p + k0][offset + 6] = 1
        V[1*p + k1][offset + 6] = 1
        W[k1*n + 1][offset + 6] = 1
        
        # Extra rank-1 terms for the third row of U (j=2)
        # Term 8: a02 * b2k0 -> c00
        U[0*m + 2][offset + 7] = 1
        V[2*p + k0][offset + 7] = 1
        W[k0*n + 0][offset + 7] = 1
        
        # Term 9: a02 * b2k1 -> c01
        U[0*m + 2][offset + 8] = 1
        V[2*p + k1][offset + 8] = 1
        W[k0*n + 1][offset + 8] = 1
        
        # Term 10: a12 * b2k0 -> c10
        U[1*m + 2][offset + 9] = 1
        V[2*p + k0][offset + 9] = 1
        W[k1*n + 0][offset + 9] = 1
        
        # Term 11: a12 * b2k1 -> c11
        U[1*m + 2][offset + 10] = 1
        V[2*p + k1][offset + 10] = 1
        W[k1*n + 1][offset + 10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
