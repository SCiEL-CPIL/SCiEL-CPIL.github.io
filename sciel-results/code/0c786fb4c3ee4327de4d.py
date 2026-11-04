"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for matrix multiplication of 3x3 by 3x4.
    This implementation uses the schoolbook decomposition, which has rank n*m*p.
    For n=3, m=3, p=4, the rank is 36.
    
    The decomposition satisfies:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if i,j,k match the term, 0 otherwise.
    Specifically, the tensor T[i,j,k] = delta_ik is decomposed.
    Wait, the matrix multiplication tensor is T[i, j, k] where C[i,k] = sum_j A[i,j] B[j,k].
    The term is A[i,j] * B[j,k].
    The output index is (i, k).
    The input indices are (i, j) for A and (j, k) for B.
    The standard form is:
    C[i, k] = sum_j A[i, j] B[j, k]
    Tensor T[i, j, k] such that T[i, j, k] = 1 if the indices form a valid path, but here we have free indices i, k and summed j.
    Actually, the tensor for multiplication is T[i, j, k] where i is row of A, j is col of A / row of B, k is col of B.
    The output is indexed by (i, k).
    The contract says:
    U shape (n*m, rank) -> indices (i, j)
    V shape (m*p, rank) -> indices (j, k)
    W shape (p*n, rank) -> indices (k, i) ? 
    "output factor follows the official trace convention W[k*n+i,r]"
    This implies W index is $k \cdot n + i$.
    So the term is $U[i, j] V[j, k] W[k, i]$.
    This matches $A[i, j] B[j, k]$.
    The schoolbook decomposition sets one term to 1 for each (i, j, k).
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
