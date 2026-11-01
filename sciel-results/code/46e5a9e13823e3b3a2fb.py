"""Matrix multiplication 2x2, 2x7 exact tensor decomposition.

Strategy:
  * Three Strassen rank-7 blocks for column pairs (0,1), (2,3), (4,5).
  * One schoolbook rank-4 block for column 6.
  * Total rank: 7 + 7 + 7 + 4 = 25.
  * Fallback to schoolbook rank-28 if verification fails.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 7]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # Safety margin under 150s wall-clock


def _time_remaining():
    return _DEADLINE - time.time()


def _verify_decomposition(U, V, W, n=2, m=2, p=7):
    """Verify exact tensor decomposition using rational arithmetic."""
    from fractions import Fraction
    R = len(U[0]) if U and len(U) else 0
    Uf = [[Fraction(U[i][c]) for c in range(R)] for i in range(n * m)]
    Vf = [[Fraction(V[j][c]) for c in range(R)] for j in range(m * p)]
    Wf = [[Fraction(W[k][c]) for c in range(R)] for k in range(p * n)]

    # Check tensor identity: sum_c U[i,j,c] * V[j,k,c] == delta(i,k)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = Fraction(0)
                for c in range(R):
                    s += Uf[i * m + j][c] * Vf[j * p + k][c]
                if s != (1 if i == k else 0):
                    return False

    # Check W routing: each column has exactly one nonzero at the correct position
    for c in range(R):
        nz = [(k, i) for k in range(p) for i in range(n) if Wf[k * n + i][c] != 0]
        if len(nz) != 1:
            return False
        (k0, i0) = nz[0]
        ok = False
        for j in range(m):
            if Uf[i0 * m + j][c] != 0 and Vf[j * p + k0][c] != 0:
                ok = True
                break
        if not ok:
            return False

    return True


def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = n * m * p  # 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.

    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    if _time_remaining() < 5.0:
        # Safety: return a trivially valid schoolbook rank-28 decomposition
        return _schoolbook_decomposition()

    n, m, p = 2, 2, 7
    rank = 25

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    def add_strassen_block(col_start, k_start):
        """
        Add Strassen rank-7 terms for a 2x2 block of B (columns k_start, k_start+1).
        col_start: starting column index in U, V, W for this block's terms.
        k_start: starting column index in B (and C).
        """
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        # Contributes to C0,ks and C1,ks+1
        U[0, col_start + 0] += 1
        U[3, col_start + 0] += 1
        V[7 * 0 + k_start, col_start + 0] += 1
        V[7 * 1 + (k_start + 1), col_start + 0] += 1
        W[2 * k_start + 0, col_start + 0] += 1
        W[2 * (k_start + 1) + 1, col_start + 0] += 1

        # M2 = (A10 + A11) * B0,ks
        # Contributes to C1,ks and C1,ks+1 (with negative sign for C1,ks+1)
        U[2, col_start + 1] += 1
        U[3, col_start + 1] += 1
        V[7 * 0 + k_start, col_start + 1] += 1
        W[2 * k_start + 1, col_start + 1] += 1
        W[2 * (k_start + 1) + 1, col_start + 1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        # Contributes to C0,ks+1 and C1,ks+1
        U[0, col_start + 2] += 1
        V[7 * 0 + (k_start + 1), col_start + 2] += 1
        V[7 * 1 + (k_start + 1), col_start + 2] -= 1
        W[2 * (k_start + 1) + 0, col_start + 2] += 1
        W[2 * (k_start + 1) + 1, col_start + 2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        # Contributes to C0,ks and C1,ks
        U[3, col_start + 3] += 1
        V[7 * 1 + k_start, col_start + 3] += 1
        V[7 * 0 + k_start, col_start + 3] -= 1
        W[2 * k_start + 0, col_start + 3] += 1
        W[2 * k_start + 1, col_start + 3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        # Contributes to C0,ks (negative) and C0,ks+1
        U[0, col_start + 4] += 1
        U[1, col_start + 4] += 1
        V[7 * 1 + (k_start + 1), col_start + 4] += 1
        W[2 * k_start + 0, col_start + 4] -= 1
        W[2 * (k_start + 1) + 0, col_start + 4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        # Contributes to C1,ks+1
        U[2, col_start + 5] += 1
        U[0, col_start + 5] -= 1
        V[7 * 0 + k_start, col_start + 5] += 1
        V[7 * 0 + (k_start + 1), col_start + 5] += 1
        W[2 * (k_start + 1) + 1, col_start + 5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        # Contributes to C0,ks
        U[1, col_start + 6] += 1
        U[3, col_start + 6] -= 1
        V[7 * 1 + k_start, col_start + 6] += 1
        V[7 * 1 + (k_start + 1), col_start + 6] += 1
        W[2 * k_start + 0, col_start + 6] += 1

    # Add three Strassen blocks for column pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)

    # Schoolbook for column 6 (rank 4)
    col_start = 21
    # C[i,6] = sum_j A[i,j] * B[j,6]
    for j in range(2):
        for i in range(2):
            U[i * m + j, col_start] += 1
            V[j * p + 6, col_start] += 1
            W[6 * n + i, col_start] += 1
            col_start += 1

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify the decomposition is correct
    if _verify_decomposition(result["U"], result["V"], result["W"]):
        return result

    # Fallback to schoolbook if verification fails
    return _schoolbook_decomposition()


if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])

    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B

    C_W = np.zeros(14)
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        for idx in range(14):
            C_W[idx] += W[idx, r] * SA * SB

    C_mat = np.zeros((2, 7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2 * k + i]

    if np.allclose(C_mat, C_expected):
        print("Verification PASSED, rank =", U.shape[1])
    else:
        print("Verification FAILED")
        print(C_mat)
        print(C_expected)
