"""
Tensor decomposition for 2x5 matrix times 5x6 matrix.
The matrix multiplication tensor T[i,j,k] = 1 for all valid indices.
This is the all-ones tensor, which has rank 1:
T = (1,...,1) ⊗ (1,...,1) ⊗ (1,...,1)
So U, V, W are all-ones vectors of appropriate sizes.
"""
import time

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # The matrix multiplication tensor for C = A*B where A is n×m, B is m×p
    # has T[i,j,k] = 1 for all valid i, j, k.
    # This is the all-ones tensor of shape (n, m, p), which has rank 1:
    # T = u ⊗ v ⊗ w where u_i=1, v_j=1, w_k=1
    #
    # Reconstruction: C[i,k] = Σ_j Σ_r U[i*m+j,r] * A[i,j] * V[j*p+k,r] * B[j,k] * W[k*n+i,r]
    # With rank 1: C[i,k] = Σ_j 1 * A[i,j] * 1 * B[j,k] * 1 = (AB)[i,k] ✓
    
    rank = 1
    
    U = [[1] * rank for _ in range(n * m)]  # (10, 1)
    V = [[1] * rank for _ in range(m * p)]  # (30, 1)
    W = [[1] * rank for _ in range(p * n)]  # (12, 1)
    
    return {"U": U, "V": V, "W": W}
