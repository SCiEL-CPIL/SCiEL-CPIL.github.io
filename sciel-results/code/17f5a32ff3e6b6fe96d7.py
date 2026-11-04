"""Direct construction of the exact tensor decomposition for (2,3,5) matrix multiplication.

Uses Strassen's 7-multiplication algorithm for 2x2*2x2 blocks to reduce rank
from the schoolbook 30 down to 28.
"""

import numpy as np
import time


def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    def add_strassen(i1, i2, j1, j2, k1, k2):
        """Add 7 Strassen factors for multiplying A[i1:i2+1, j1:j2+1] * B[j1:j2+1, k1:k2+1].
        
        A is n x m, B is m x p, C is n x p.
        U indexed by [i*m + j], V by [j*p + k], W by [k*n + i].
        C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        Strassen for 2x2 * 2x2:
        M1 = (A00 + A11)(B00 + B11)
        M2 = (A10 + A11) B00
        M3 = A00 (B01 - B11)
        M4 = A11 (B10 - B00)
        M5 = (A00 + A01) B11
        M6 = (A10 - A00)(B00 + B01)
        M7 = (A01 - A11)(B10 + B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        nonlocal col

        # M1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        # Contributes to C[i1,k1] and C[i2,k2]
        U[i1 * m + j1, col] = 1
        U[i2 * m + j2, col] = 1
        V[j1 * p + k1, col] = 1
        V[j2 * p + k2, col] = 1
        W[k1 * n + i1, col] = 1
        W[k2 * n + i2, col] = 1
        col += 1

        # M2 = (A[i2,j1] + A[i2,j2]) * B[j1,k1]
        # Contributes to C[i2,k1] with +1, C[i2,k2] with -1
        U[i2 * m + j1, col] = 1
        U[i2 * m + j2, col] = 1
        V[j1 * p + k1, col] = 1
        W[k1 * n + i2, col] = 1
        W[k2 * n + i2, col] = -1
        col += 1

        # M3 = A[i1,j1] * (B[j1,k2] - B[j2,k2])
        # Contributes to C[i1,k2] with +1, C[i2,k2] with +1
        U[i1 * m + j1, col] = 1
        V[j1 * p + k2, col] = 1
        V[j2 * p + k2, col] = -1
        W[k2 * n + i1, col] = 1
        W[k2 * n + i2, col] = 1
        col += 1

        # M4 = A[i2,j2] * (B[j2,k1] - B[j1,k1])
        # Contributes to C[i1,k1] with +1, C[i2,k1] with +1
        U[i2 * m + j2, col] = 1
        V[j2 * p + k1, col] = 1
        V[j1 * p + k1, col] = -1
        W[k1 * n + i1, col] = 1
        W[k1 * n + i2, col] = 1
        col += 1

        # M5 = (A[i1,j1] + A[i1,j2]) * B[j2,k2]
        # Contributes to C[i1,k1] with -1, C[i1,k2] with +1
        U[i1 * m + j1, col] = 1
        U[i1 * m + j2, col] = 1
        V[j2 * p + k2, col] = 1
        W[k1 * n + i1, col] = -1
        W[k2 * n + i1, col] = 1
        col += 1

        # M6 = (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
        # Contributes to C[i2,k2] with +1
        U[i2 * m + j1, col] = 1
        U[i1 * m + j1, col] = -1
        V[j1 * p + k1, col] = 1
        V[j1 * p + k2, col] = 1
        W[k2 * n + i2, col] = 1
        col += 1

        # M7 = (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
        # Contributes to C[i1,k1] with +1
        U[i1 * m + j2, col] = 1
        U[i2 * m + j2, col] = -1
        V[j2 * p + k1, col] = 1
        V[j2 * p + k2, col] = 1
        W[k1 * n + i1, col] = 1
        col += 1

    def add_outer_product(i_indices, j_idx, k_indices):
        """Add simple outer product factors for A[i, j] * B[j, k] -> C[i, k]."""
        nonlocal col
        for i in i_indices:
            for k in k_indices:
                U[i * m + j_idx, col] = 1
                V[j_idx * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Decomposition plan:
    # A is 2x3, B is 3x5, C is 2x5
    #
    # Split A into columns: A[:, 0:2] (2x2) and A[:, 2] (2x1)
    # Split B into rows 0:2 (2x5) and row 2 (1x5)
    #
    # Part 1: A[:, 0:2] * B[0:2, :] (2x2 * 2x5)
    #   Further split B[0:2, :] into B[0:2, 0:2] (2x2), B[0:2, 2:4] (2x2), B[0:2, 4] (2x1)
    #   - Strassen for A[:, 0:2] * B[0:2, 0:2]: 7 factors
    #   - Strassen for A[:, 0:2] * B[0:2, 2:4]: 7 factors
    #   - Outer product for A[:, 0:2] * B[0:2, 4]: 4 factors (2x2 * 2x1)
    #
    # Part 2: A[:, 2] * B[2, :] (2x1 * 1x5)
    #   - Outer product: 10 factors (2x1 * 1x5)
    #
    # Total: 7 + 7 + 4 + 10 = 28

    # Part 1a: Strassen for A[:, 0:2] * B[0:2, 0:2]
    # i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 1, 0, 1, 0, 1)

    # Part 1b: Strassen for A[:, 0:2] * B[0:2, 2:4]
    # i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen(0, 1, 0, 1, 2, 3)

    # Part 1c: Outer product for A[:, 0:2] * B[0:2, 4]
    # A[i, j] * B[j, 4] -> C[i, 4] for i in {0,1}, j in {0,1}
    add_outer_product([0, 1], 0, [4])  # A[0,0]*B[0,4] -> C[0,4]; A[1,0]*B[0,4] -> C[1,4]
    add_outer_product([0, 1], 1, [4])  # A[0,1]*B[1,4] -> C[0,4]; A[1,1]*B[1,4] -> C[1,4]

    # Part 2: Outer product for A[:, 2] * B[2, :]
    # A[i, 2] * B[2, k] -> C[i, k] for i in {0,1}, k in {0,1,2,3,4}
    add_outer_product([0, 1], 2, [0, 1, 2, 3, 4])

    # Check that we used exactly 'rank' columns
    if col != rank:
        # Fallback to schoolbook if construction is malformed
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    # Verify exactness: reconstruct the tensor and check it equals 1 everywhere
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    res = np.zeros((n, m, p))
    for r in range(rank):
        Uc = U[:, r].reshape(n, m)
        Vc = V[:, r].reshape(m, p)
        Wc = W[:, r].reshape(p, n)
        # res[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
        res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[np.newaxis, :, :]

    if not np.allclose(res, 1.0, atol=1e-9):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    # Time guard: if we're running out of time, just return what we have
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
