import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 10.0

    n, m, p = DIMENSIONS
    rank = 7

    # U: (n*m, rank) -> (4, 7)
    # V: (m*p, rank) -> (4, 7)
    # W: (p*n, rank) -> (4, 7)
    # Indices:
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i
    # Tensor element T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Must equal 1 for all i, j, k in {0, 1}

    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # r=0: M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0

    # r=1: M2 = (A10 + A11) * B00
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[1, 1] = 1.0; W[3, 1] = -1.0

    # r=2: M3 = A00 * (B01 - B11)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[2, 2] = 1.0; W[3, 2] = 1.0

    # r=3: M4 = A11 * (B10 - B00)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[1, 3] = 1.0

    # r=4: M5 = (A00 + A01) * B11
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[2, 4] = 1.0

    # r=5: M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0

    # r=6: M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
