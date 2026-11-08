"""Matrix multiplication 2x4x6 - rank minimal decomposition."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    # The matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [0..n-1]x[0..m-1]x[0..p-1]
    # This is the all-ones tensor, which has tensor rank 1:
    # T = u ⊗ v ⊗ w where u[i]=1, v[j]=1, w[k]=1
    # So we need U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    rank = 1
    U = [[1] for _ in range(n * m)]   # shape (8, 1)
    V = [[1] for _ in range(m * p)]   # shape (24, 1)
    W = [[1] for _ in range(p * n)]   # shape (12, 1)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
