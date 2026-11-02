"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The tensor for standard matrix multiplication A(nxm) * B(mxk) -> C(nxk)
    # has entries T[i, j, k] = 1 for all valid i, j, k.
    # This is because C[i, k] = sum_j A[i, j] * B[j, k].
    # Every term A[i, j] * B[j, k] contributes with coefficient 1.
    # Thus, the tensor is the all-ones tensor of size n x m x p.
    # The rank of the all-ones tensor is 1.
    # We can decompose it as:
    # U[i*m + j, 0] = 1
    # V[j*p + k, 0] = 1
    # W[k*n + i, 0] = 1
    # Then sum_c U[...] V[...] W[...] = 1 * 1 * 1 = 1.
    
    rank = 1
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill with 1s
    for i in range(n * m):
        U[i][0] = 1
    for j in range(m * p):
        V[j][0] = 1
    for k in range(p * n):
        W[k][0] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
