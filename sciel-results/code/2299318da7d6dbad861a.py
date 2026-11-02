"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-98 decomposition for 4x4 * 4x8 matrix multiplication using Strassen.
    This is better than schoolbook rank 128.
    """
    n, m, p = 4, 4, 8
    
    # Strassen decomposition for 2x2 * 2x2 (rank 7)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=m=p=2
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],  # (i=0,j=0)
        [0, 0, 0, 0, 1, 0, 1],   # (i=0,j=1)
        [1, 1, 0, 0, 0, 1, 0],   # (i=1,j=0)
        [1, 1, 0, 1, 0, 0, -1]   # (i=1,j=1)
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],  # (j=0,k=0)
        [0, 0, 1, 0, 0, 1, 0],   # (j=0,k=1)
        [0, 0, 0, 1, 0, 0, 1],   # (j=1,k=0)
        [1, 0, -1, 0, 1, 0, 1]   # (j=1,k=1)
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],  # (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],   # (k=0,i=1)
        [0, 1, 0, 1, 0, 0, 0],   # (k=1,i=0)
        [1, -1, 1, 0, 0, 1, 1]   # (k=1,i=1)
    ]
    
    # Kronecker product to get (4,4,4) decomposition (rank 49)
    # (4,4,4) = (2,2,2) ⊗ (2,2,2)
    rank_s = 7
    rank_444 = rank_s * rank_s  # 49
    
    U444 = [[0] * rank_444 for _ in range(16)]  # (4*4, 49)
    V444 = [[0] * rank_444 for _ in range(16)]  # (4*4, 49)
    W444 = [[0] * rank_444 for _ in range(16)]  # (4*4, 49)
    
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    idx_U = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                    for r1 in range(rank_s):
                        for r2 in range(rank_s):
                            r = r1 * rank_s + r2
                            U444[idx_U][r] = S_U[i1 * 2 + j1][r1] * S_U[i2 * 2 + j2][r2]
    
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    idx_V = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                    for r1 in range(rank_s):
                        for r2 in range(rank_s):
                            r = r1 * rank_s + r2
                            V444[idx_V][r] = S_V[j1 * 2 + k1][r1] * S_V[j2 * 2 + k2][r2]
    
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    idx_W = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                    for r1 in range(rank_s):
                        for r2 in range(rank_s):
                            r = r1 * rank_s + r2
                            W444[idx_W][r] = S_W[k1 * 2 + i1][r1] * S_W[k2 * 2 + i2][r2]
    
    # Now build (4,4,8) from two (4,4,4) decompositions
    rank_448 = 2 * rank_444  # 98
    
    U448 = [[0] * rank_448 for _ in range(16)]  # (16, 98)
    V448 = [[0] * rank_448 for _ in range(32)]  # (32, 98)
    W448 = [[0] * rank_448 for _ in range(32)]  # (32, 98)
    
    # U is shared
    for i in range(16):
        for r in range(rank_444):
            U448[i][r] = U444[i][r]
            U448[i][r + rank_444] = U444[i][r]
    
    # V: first 16 rows for k=0..3, next 16 rows for k=4..7
    for j in range(4):
        for k in range(4):
            idx = j * 8 + k
            for r in range(rank_444):
                V448[idx][r] = V444[j * 4 + k][r]
                V448[idx][r + rank_444] = 0
            idx2 = j * 8 + (k + 4)
            for r in range(rank_444):
                V448[idx2][r] = 0
                V448[idx2][r + rank_444] = V444[j * 4 + k][r]
    
    # W: first 16 rows for k=0..3, next 16 rows for k=4..7
    for k in range(4):
        for i in range(4):
            idx = k * 4 + i
            for r in range(rank_444):
                W448[idx][r] = W444[k * 4 + i][r]
                W448[idx][r + rank_444] = 0
            idx2 = (k + 4) * 4 + i
            for r in range(rank_444):
                W448[idx2][r] = 0
                W448[idx2][r + rank_444] = W444[k * 4 + i][r]
    
    return {"U": U448, "V": V448, "W": W448}
# EVOLVE-BLOCK-END
