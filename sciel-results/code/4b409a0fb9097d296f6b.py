"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
"""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the factor matrices U, V, W for the schoolbook decomposition of
    multiplying a 3x3 matrix by a 3x5 matrix.
    
    The rank is n*m*p = 3*3*5 = 45.
    U has shape (n*m, rank) = (9, 45)
    V has shape (m*p, rank) = (15, 45)
    W has shape (p*n, rank) = (15, 45)
    
    The decomposition satisfies:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k} (conceptually, 
    actually it reconstructs the product C[i,k] = sum_j A[i,j]B[j,k])
    
    Specifically, the tensor T_{i,j,k} for matrix multiplication (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    is decomposed as sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} ? 
    Wait, the standard tensor for matrix multiplication M_{n,m,p} has indices (i, j, k) 
    where i is row of A, j is col of A / row of B, k is col of B.
    The factors are usually defined such that:
    (A * B)_{i, k} = sum_j A_{i, j} B_{j, k}
    
    The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k (e_i \otimes e_j \otimes e_k)^* ? No.
    
    Let's stick to the seed program's definition which is standard:
    U[i*m+j, c]
    V[j*p+k, c]
    W[k*n+i, c]
    
    For each term c corresponding to (i,j,k):
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1
    
    This ensures that the product sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} 
    is 1 if the indices match the term c, and 0 otherwise.
    This reconstructs the identity tensor for the multiplication operation.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
