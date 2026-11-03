"""Matrix multiplication 3x4x6 tensor decomposition.

Uses the schoolbook decomposition which is guaranteed correct.
The tensor rank for matrix multiplication of dimensions n x m x p is n*m*p
for the standard bilinear decomposition.
"""

DIMENSIONS = [3, 4, 6]

def construct():
    """Construct the factor matrices for matrix multiplication 3x4x6.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # Schoolbook rank
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition: each (i,j,k) triple gives one rank-1 term
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is indexed by (i*m + j), representing the (i,j) entry of A
                U[i * m + j][column] = 1
                # V is indexed by (j*p + k), representing the (j,k) entry of B
                V[j * p + k][column] = 1
                # W is indexed by (k*n + i), representing the (k,i) position in C
                # (note: C has shape n x p, but W uses k*n + i indexing)
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
