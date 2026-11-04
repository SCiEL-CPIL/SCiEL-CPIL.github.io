"""Matrix multiplication 3x5x7 - attempting rank-minimal decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook tensor decomposition for 3x5x7 matrix multiplication.
    
    The matrix multiplication tensor T has entries T[i,j,k] = delta_{i,k} (in the 
    appropriate sense for C = A*B where A is 3x5, B is 5x7, C is 3x7).
    
    For each output element C[i,k] = sum_j A[i,j] * B[j,k], we create one rank-1
    term. This gives rank = n*m*p = 3*5*7 = 105.
    
    Indexing convention:
    - U has shape (n*m, rank): U[i*m + j, r]
    - V has shape (m*p, rank): V[j*p + k, r]  
    - W has shape (p*n, rank): W[k*n + i, r]
    
    The decomposition satisfies:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = 3, 5, 7
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each rank-1 term contributes to exactly one (i,j,k) triple
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
