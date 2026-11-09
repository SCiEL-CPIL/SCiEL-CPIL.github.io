"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal decomposition for 2x4 * 4x6.
    # The tensor is a direct sum over k of 8 independent terms (rank 2 each).
    # Total rank = 6 * 8 = 48.
    n, m, p = DIMENSIONS
    # Rank for the 2x4 * 4x1 case is 8 (since each of the 4 j's contributes 2).
    # For 2x4 * 4x6, it's 6 * 8 = 48.
    # However, we can just use schoolbook which is 2*4*6 = 48.
    # Let's verify if we can do better.
    # No, the disjoint support in V (indexed by j) prevents sharing across j.
    # Within each j, the 2x1 * 1x6 tensor has rank 2.
    # So total rank is 4 * 6 * 2 = 48.
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # For each j in 0..3:
    #   For each k in 0..5:
    #     Term 1: handles A[0,j] * B[j,k] -> C[k,0] and A[0,j] * B[j,k] * 0? No.
    #     Let's use the standard rank-2 decomposition for A[0,j]*B[j,k] -> C[k,0], C[k,1].
    #     Actually, we need to produce C[k,0] = A[0,j]*B[j,k] and C[k,1] = A[1,j]*B[j,k].
    #     Rank 2 terms:
    #     T1: U[0*m+j] = 1, V[j*p+k] = 1, W[k*n+0] = 1, W[k*n+1] = 0
    #          Wait, this gives A[0,j]*B[j,k] to C[k,0]. What about C[k,1]?
    #     T2: U[0*m+j] = 0, V[j*p+k] = 1, W[k*n+0] = 0, W[k*n+1] = 1
    #          This gives 0 to C[k,0] and ... wait, U is 0 so it contributes 0.
    #     
    #     We need:
    #     C[k,0] += A[0,j] B[j,k]
    #     C[k,1] += A[1,j] B[j,k]
    #     
    #     Term 1: U[0*m+j]=1, U[1*m+j]=0. V[j*p+k]=1. W[k*n+0]=1, W[k*n+1]=0.
    #              Contribution: A[0,j] * B[j,k] * 1 to C[k,0]. Correct.
    #     Term 2: U[0*m+j]=0, U[1*m+j]=1. V[j*p+k]=1. W[k*n+0]=0, W[k*n+1]=1.
    #              Contribution: A[1,j] * B[j,k] * 1 to C[k,1]. Correct.
    #     
    #     So for each (j,k), we use 2 columns.
    #     Total columns = 4 * 6 * 2 = 48.
    
    for j in range(m):
        for k in range(p):
            # Column for C[k,0]
            U[0 * m + j][col] = 1
            U[1 * m + j][col] = 0
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            W[k * n + 1][col] = 0
            col += 1
            
            # Column for C[k,1]
            U[0 * m + j][col] = 0
            U[1 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 0
            W[k * n + 1][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
