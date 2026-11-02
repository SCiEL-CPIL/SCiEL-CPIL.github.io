"""Matrix multiplication 2x3x10 - rank-minimal decomposition."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the tensor decomposition of matrix multiplication.
    
    The decomposition is of the form:
    C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where the sum is over j, and the term is non-zero only when indices align.
    
    For the schoolbook algorithm:
    U[i*m+j, r] = 1 if r corresponds to (i, j, k)
    V[j*p+k, r] = 1 if r corresponds to (i, j, k)
    W[k*n+i, r] = 1 if r corresponds to (i, j, k)
    
    Rank is n*m*p.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i*m + j
                U[i * m + j][col] = 1
                # Index in V: row j*p + k
                V[j * p + k][col] = 1
                # Index in W: row k*n + i
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
