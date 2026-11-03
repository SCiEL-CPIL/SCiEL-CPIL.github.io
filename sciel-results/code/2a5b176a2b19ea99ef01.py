import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # The tensor for matrix multiplication C_ik = sum_j A_ij B_jk
    # is T_ijk = 1 for all i, j, k.
    # This is a rank-1 tensor: T = u (v (w)) with u_i=1, v_j=1, w_k=1.
    # U has shape (n*m, rank). V has shape (m*p, rank). W has shape (p*n, rank).
    # For rank 1:
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
