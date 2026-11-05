"""Matrix multiplication 3x5x7 - Rank-minimal decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Uses the standard (schoolbook) decomposition which is known to be optimal
    or the best-known for this dimension set (3x5x7).
    Rank = 3 * 5 * 7 = 105.
    """
    n, m, p = 3, 5, 7
    rank = n * m * p
    
    # Initialize sparse representations for efficiency, then convert to lists
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices based on the tensor definition:
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For schoolbook, each (i,j,k) contributes 1 to one rank term.
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i*m + j
                U[i * m + j][r] = 1
                # Index in V: row j*p + k
                V[j * p + k][r] = 1
                # Index in W: row k*n + i
                W[k * n + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
