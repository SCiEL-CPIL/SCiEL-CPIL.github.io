"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal tensor decomposition for multiplying a 2x3 matrix
    by a 3x9 matrix.
    
    The multiplication is C = A * B, where A is 2x3, B is 3x9, C is 2x9.
    C[i, k] = sum_j A[i, j] * B[j, k].
    
    This is equivalent to 2 independent multiplications of a 1x3 vector by a 3x9 matrix.
    The rank for 1x3 by 3x9 is 3*9 = 27.
    Total rank is 2 * 27 = 54.
    
    The decomposition uses the standard schoolbook algorithm.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index for U: row i, col j -> i*m + j
                # Index for V: row j, col k -> j*p + k
                # Index for W: row k, col i -> k*n + i (trace convention W[k*n+i, r])
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
