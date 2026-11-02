"""
Program for matrix multiplication 2,2,7.
Uses a rank-25 decomposition based on Strassen's algorithm for 2x2 blocks.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 7]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin before 120s limit

def _time_remaining():
    return _DEADLINE - time.time()

# EVOLVE-BLOCK-START
def _build_rank25_decomposition(n, m, p):
    """Build the standard rank-25 decomposition using Strassen blocks."""
    rank = 25
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start + 0] += 1
        U[3, col_start + 0] += 1
        V[7 * 0 + k_start, col_start + 0] += 1
        V[7 * 1 + (k_start + 1), col_start + 0] += 1
        W[2 * k_start + 0, col_start + 0] += 1
        W[2 * (k_start + 1) + 1, col_start + 0] += 1

        # M2 = (A10 + A11) * B0,ks
        U[2, col_start + 1] += 1
        U[3, col_start + 1] += 1
        V[7 * 0 + k_start, col_start + 1] += 1
        W[2 * k_start + 1, col_start + 1] += 1
        W[2 * (k_start + 1) + 1, col_start + 1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start + 2] += 1
        V[7 * 0 + (k_start + 1), col_start + 2] += 1
        V[7 * 1 + (k_start + 1), col_start + 2] -= 1
        W[2 * (k_start + 1) + 0, col_start + 2] += 1
        W[2 * (k_start + 1) + 1, col_start + 2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start + 3] += 1
        V[7 * 1 + k_start, col_start + 3] += 1
        V[7 * 0 + k_start, col_start + 3] -= 1
        W[2 * k_start + 0, col_start + 3] += 1
        W[2 * k_start + 1, col_start + 3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start + 4] += 1
        U[1, col_start + 4] += 1
        V[7 * 1 + (k_start + 1), col_start + 4] += 1
        W[2 * k_start + 0, col_start + 4] -= 1
        W[2 * (k_start + 1) + 0, col_start + 4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start + 5] += 1
        U[0, col_start + 5] -= 1
        V[7 * 0 + k_start, col_start + 5] += 1
        V[7 * 0 + (k_start + 1), col_start + 5] += 1
        W[2 * (k_start + 1) + 1, col_start + 5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start + 6] += 1
        U[3, col_start + 6] -= 1
        V[7 * 1 + k_start, col_start + 6] += 1
        V[7 * 1 + (k_start + 1), col_start + 6] += 1
        W[2 * k_start + 0, col_start + 6] += 1

    # Strassen for columns 0,1
    add_strassen(0, 0)
    # Strassen for columns 2,3
    add_strassen(7, 2)
    # Strassen for columns 4,5
    add_strassen(14, 4)
    # Schoolbook for column 6
    cs = 21
    U[0, cs + 0] += 1
    V[7 * 0 + 6, cs + 0] += 1
    W[2 * 6 + 0, cs + 0] += 1

    U[1, cs + 1] += 1
    V[7 * 1 + 6, cs + 1] += 1
    W[2 * 6 + 0, cs + 1] += 1

    U[2, cs + 2] += 1
    V[7 * 0 + 6, cs + 2] += 1
    W[2 * 6 + 1, cs + 2] += 1

    U[3, cs + 3] += 1
    V[7 * 1 + 6, cs + 3] += 1
    W[2 * 6 + 1, cs + 3] += 1

    return U, V, W


def _build_rank28_schoolbook(n, m, p):
    """Fallback rank-28 schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition correctness."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.max(np.abs(T_recon - T)) <= tol


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    n, m, p = 2, 2, 7

    # Safety: if very little time remains, return schoolbook
    if _time_remaining() < 2.0:
        U, V, W = _build_rank28_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the rank-25 decomposition
    U, V, W = _build_rank25_decomposition(n, m, p)

    # Verify the solution
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U, V, W = _build_rank28_schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
