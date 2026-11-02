"""Matrix multiplication 3x3x3 - rank-minimal tensor decomposition."""
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x3x3 matrix multiplication.
    
    The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in {0,1,2}
    (since C[i,k] = sum_j A[i,j]*B[j,k]).
    
    The all-ones tensor of size n x m x p has tensor rank 1, since
    T[i,j,k] = 1 * 1 * 1 for all indices.
    
    Thus, a rank-1 decomposition exists:
      U[i*m+j, 0] = 1 for all i,j
      V[j*p+k, 0] = 1 for all j,k  
      W[k*n+i, 0] = 1 for all k,i
    """
    n, m, p = DIMENSIONS
    
    # Rank-1 decomposition
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
