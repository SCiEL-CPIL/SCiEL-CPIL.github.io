"""
Exact tensor decomposition for 3x5 matrix by 5x6 matrix.
This implementation provides the standard schoolbook decomposition with rank n*m*p.
"""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    of the tensor decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices as lists of lists of integers (0s)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate through all indices i (0..n-1), j (0..m-1), k (0..p-1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i, col j -> i*m + j
                # V index: row j, col k -> j*p + k
                # W index: row k, col i -> k*n + i
                # Following the trace convention W[k*n + i, r]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
