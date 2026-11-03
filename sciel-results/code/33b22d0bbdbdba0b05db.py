"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook decomposition of rank n*m*p.
    
    The matrix multiplication tensor T_{i,j,k} satisfies:
    T_{i,j,k} = 1 for all valid (i,j,k), meaning the product C[i,k] = sum_j A[i,j]*B[j,k]
    is recovered by summing over rank terms:
    C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  (appropriately indexed)
    
    Actually, the standard bilinear decomposition for matrix multiplication is:
    T_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r
    where U, V, W are the factor matrices.
    
    For the schoolbook algorithm, each (i,j,k) tuple contributes one rank term:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each c corresponding to (i,j,k).
    
    This gives rank = n*m*p = 3*5*5 = 75.
    
    While not rank-minimal, it is a valid exact decomposition that always works.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
