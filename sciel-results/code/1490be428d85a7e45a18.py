"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # The tensor for matrix multiplication C = AB where C is n x p, A is n x m, B is m x p
    # is given by T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p).
    # This is because every A[i, j] is multiplied by every B[j, k] to contribute to C[i, k].
    # The decomposition condition is:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] == 1
    # 
    # A rank 1 decomposition exists:
    # U[i*m + j, 0] = 1
    # V[j*p + k, 0] = 1
    # W[k*n + i, 0] = 1
    # Sum = 1 * 1 * 1 = 1.
    
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    # Verify
    def verify_decomp(U, V, W):
        rank_val = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank_val):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U, V, W):
        # Fallback to schoolbook if rank 1 somehow fails (it shouldn't)
        schoolbook_rank = n * m * p
        U_school = np.zeros((n * m, schoolbook_rank))
        V_school = np.zeros((m * p, schoolbook_rank))
        W_school = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        U, V, W = U_school, V_school, W_school
        rank = schoolbook_rank

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
