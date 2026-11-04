"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the standard schoolbook tensor decomposition for matrix multiplication.
    For multiplying an (n x m) matrix by an (m x p) matrix, the rank is n*m*p.
    
    The decomposition is defined by:
    U[i*m + j, c] = 1
    V[j*p + k, c] = 1
    W[k*n + i, c] = 1
    where c corresponds to the unique combination of (i, j, k).
    
    The output factor W follows the convention W[k*n + i, r].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i of A, col j -> i*m + j
                U[i * m + j][column] = 1
                # V index: row j of B, col k -> j*p + k
                V[j * p + k][column] = 1
                # W index: row k of C, col i -> k*n + i (per task convention)
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
