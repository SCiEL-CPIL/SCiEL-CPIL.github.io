"""Rewritten program for Matrix Multiplication 3,5,7."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook decomposition for matrix multiplication of dimensions n x m and m x p.
    This results in a rank of n * m * p.
    
    The decomposition is defined by:
    C[i, k] = sum_{j} A[i, j] * B[j, k]
    
    In tensor form T[i, j, k] = 1 for all valid i, j, k.
    The rank-1 components are u_{ijk} = e_i (in A) \otimes e_j (in B) \otimes e_k (in C).
    
    Matrices:
    U: shape (n*m, rank). Row i*m+j has 1 at column corresponding to (i,j,k).
    V: shape (m*p, rank). Row j*p+k has 1 at column corresponding to (i,j,k).
    W: shape (p*n, rank). Row k*n+i has 1 at column corresponding to (i,j,k).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists for compatibility with the expected return type
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping:
                # U row: i*m + j
                # V row: j*p + k
                # W row: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
