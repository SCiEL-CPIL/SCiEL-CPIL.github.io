"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or low-rank) exact tensor decomposition
    for multiplying a 4x4 matrix by a 4x5 matrix.
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes:
        U: (n*m, rank) = (16, rank)
        V: (m*p, rank) = (20, rank)  
        W: (p*n, rank) = (20, rank)
        
    The decomposition satisfies: for all i in [0,n), j in [0,p),
    (A*B)[i,j] = sum_{k=0}^{m-1} A[i,k] * B[k,j]
    can be expressed as sum over rank terms using U, V, W.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]  # (16, 80)
    V = [[0] * rank for _ in range(m * p)]  # (20, 80)
    W = [[0] * rank for _ in range(p * n)]  # (20, 80)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] = 1
                # V[j*p + k, column] = 1  
                # W[k*n + i, column] = 1
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
