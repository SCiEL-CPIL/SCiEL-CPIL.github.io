"""Matrix multiplication 2x3x3: schoolbook rank-18 exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """Return the exact rank-18 (schoolbook) tensor decomposition for 2x3x3 matrix multiplication.

    The multiplication tensor has T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    We output factor matrices U, V, W such that:
        U has shape (n*m, rank) = (6, 18)
        V has shape (m*p, rank) = (9, 18)
        W has shape (p*n, rank) = (6, 18)
    with the trace convention W[k*n + i, r] (i.e., output index is (k, i) mapped to k*n+i).

    For each (i, j, k) triple, column r corresponds to that term:
        U[i*m + j, r] = 1
        V[j*p + k, r] = 1
        W[k*n + i, r] = 1
    All other entries are 0.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 18

    # Build using integer arrays for exactness
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Convert to nested lists of Python ints (exact)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
