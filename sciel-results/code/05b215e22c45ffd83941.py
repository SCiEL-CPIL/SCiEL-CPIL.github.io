import time

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Returns the factor matrices U, V, W for the tensor decomposition of matrix multiplication
    of a 3x4 matrix by a 4x7 matrix.
    
    The decomposition is the standard schoolbook algorithm, which has rank n*m*p.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    The tensor T is defined such that T[i, j, k] corresponds to the multiplication of A[i,j] and B[j,k] to produce C[i,k].
    The decomposition is T = sum_c U[:,c] \otimes V[:,c] \otimes W[:,c].
    
    Specifically, for the schoolbook algorithm:
    Each term c corresponds to a unique (i, j, k) triple.
    U[i*m + j, c] = 1
    V[j*p + k, c] = 1
    W[k*n + i, c] = 1
    
    Note on W indexing: The task specifies W[k*n+i, r].
    Standard matrix multiplication C[i,k] = sum_j A[i,j] B[j,k].
    The tensor index for the result is usually (i, k).
    The prompt says "output factor follows the official trace convention W[k*n+i,r]".
    This implies the index in W is $idx = k \cdot n + i$.
    In the schoolbook term for (i,j,k):
    We need U to select A[i,j], V to select B[j,k], and W to select C[i,k].
    So W index should be $k \cdot n + i$.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists for standard compliance, though numpy is available
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (as per prompt convention)
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
