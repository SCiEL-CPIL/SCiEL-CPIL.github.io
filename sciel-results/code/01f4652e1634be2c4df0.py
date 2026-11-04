"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct the rank-minimal exact decomposition for 2x4 times 4x6.
    
    For matrix multiplication of (n x m) * (m x p), the tensor rank is n*m*p
    for the standard schoolbook algorithm. For n=2, m=4, p=6, this gives rank 48.
    
    The decomposition uses the trace convention:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Each term in the sum corresponds to a rank-1 tensor with:
    - U[i*m+j, c] = 1 if c = (i,j,k)
    - V[j*p+k, c] = 1 if c = (i,j,k)  
    - W[k*n+i, c] = 1 if c = (i,j,k)
    
    This gives an exact decomposition with rank = n*m*p = 48.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 2*4*6 = 48
    
    # Initialize factor matrices
    U = [[0.0] * rank for _ in range(n * m)]  # (8, 48)
    V = [[0.0] * rank for _ in range(m * p)]  # (24, 48)
    W = [[0.0] * rank for _ in range(p * n)]  # (12, 48)
    
    # Each column c corresponds to a unique (i, j, k) triple
    col = 0
    for i in range(n):      # row of A, row of C
        for j in range(m):  # col of A, row of B
            for k in range(p):  # col of B, col of C
                # U index: i*m + j
                U[i * m + j][col] = 1.0
                # V index: j*p + k
                V[j * p + k][col] = 1.0
                # W index: k*n + i (trace convention: W[k*n+i, r])
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
