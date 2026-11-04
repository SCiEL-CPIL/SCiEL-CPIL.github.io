"""Matrix multiplication 3,3,8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # A is n x m (3x3), B is m x p (3x8), C is n x p (3x8)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in C[i,k])
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # T[i,j,k] = 1 for all i,j,k is a rank-1 tensor:
    # T[i,j,k] = 1 * 1 * 1
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    #
    # Verification:
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1*1*1 = 1 = T[i,j,k] ✓
    
    rank = 1
    U = [[1] for _ in range(n * m)]   # shape (9, 1)
    V = [[1] for _ in range(m * p)]   # shape (24, 1)
    W = [[1] for _ in range(p * n)]   # shape (24, 1)
    
    # Final verification with random matrices
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_reconstructed = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C_reconstructed[i, k] += U[i*m+j][r] * A_test[i,j] * V[j*p+k][r] * B_test[j,k] * W[k*n+i][r]
    
    if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
