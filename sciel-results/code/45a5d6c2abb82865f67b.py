"""Find a low-rank exact tensor decomposition for 2x4 * 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    # A is n×m = 2×4, B is m×p = 4×4, C is n×p = 2×4
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # U shape: (n*m, rank) = (8, rank), indexed by i*m+j
    # V shape: (m*p, rank) = (16, rank), indexed by j*p+k
    # W shape: (p*n, rank) = (8, rank), indexed by k*n+i (trace convention)

    # Rank-28 construction:
    # For k=0,1,2: 8 separate terms each (A[i,j]*B[j,k])
    # For k=3: 4 combined terms ((A[0,j]+A[1,j])*B[j,3])
    # Total: 24 + 4 = 28

    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    t = 0
    # For k = 0, 1, 2: standard 8 terms each
    for k in range(3):
        for i in range(n):
            for j in range(m):
                U[i * m + j, t] = 1
                V[j * p + k, t] = 1
                W[k * n + i, t] = 1
                t += 1

    # For k = 3: 4 combined terms, one per j
    for j in range(m):
        for i in range(n):
            U[i * m + j, t] = 1
        V[j * p + 3, t] = 1
        for i in range(n):
            W[3 * n + i, t] = 1
        t += 1

    # Convert to lists
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
