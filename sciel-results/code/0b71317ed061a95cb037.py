"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 7 matrix.
    
    The rank of the matrix multiplication tensor M_{n,m,p} is n*m*p when 
    p >= n + m - 1. Here n=2, m=3, p=7, and 7 >= 2+3-1=4, so the rank is 42.
    
    Returns:
        A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
        U shape: (n*m, rank)
        V shape: (m*p, rank)
        W shape: (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U[i*m + j, r] corresponds to coefficient for A[i,j]
    # V[j*p + k, r] corresponds to coefficient for B[j,k]
    # W[k*n + i, r] corresponds to coefficient for C[i,k] (trace convention W[k*n+i])
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term in the schoolbook expansion is A[i,j] * B[j,k] * 1
                # U selects A[i,j]
                U[i * m + j][col] = 1
                # V selects B[j,k]
                V[j * p + k][col] = 1
                # W selects C[i,k]
                # The convention is W[k*n + i, r]
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
