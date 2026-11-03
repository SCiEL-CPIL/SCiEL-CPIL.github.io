"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition of the matrix multiplication tensor
    for multiplying a 3x3 matrix by a 3x5 matrix.
    
    The rank is n*m*p = 3*3*5 = 45.
    U has shape (n*m, rank) = (9, 45)
    V has shape (m*p, rank) = (15, 45)
    W has shape (p*n, rank) = (15, 45)
    
    The decomposition satisfies:
    sum_{r=0}^{R-1} U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    Wait, the tensor T_{i,j,k} for (A_{i,j} * B_{j,k}) is 1 if indices match in the product?
    The matrix multiplication tensor T is defined such that (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    The tensor T_{i,j,k} has 1 at (i,j,k) if we are computing the term A_{i,j} B_{j,k}.
    Actually, the standard tensor decomposition for matmul M_{n,m,p} is:
    T = sum_{r=1}^R U_r \otimes V_r \otimes W_r
    where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}.
    The schoolbook algorithm uses R = n*m*p.
    For each (i, j, k), we have a term.
    U_{i*m+j, r} = 1 if r corresponds to (i,j,k)
    V_{j*p+k, r} = 1 if r corresponds to (i,j,k)
    W_{k*n+i, r} = 1 if r corresponds to (i,j,k)
    
    This matches the current implementation.
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
