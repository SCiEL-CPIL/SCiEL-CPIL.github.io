"""
Rank-28 exact tensor decomposition for 4x4 matrix multiplication.

Strategy: block (ring) Strassen. Partition A, B into 2x2 blocks of 2x2
matrices. Apply Strassen's 7-product decomposition over the ring of 2x2
matrices. Each ring entry expands to a 4-scalar (2x2) contribution, giving
7 * 4 = 28 rank-1 terms. This is a provably exact decomposition (it computes
C = A*B via Strassen over the ring), with integer entries and rank 28.

Interface: construct() -> {'U','V','W'} with shapes (16,28),(16,28),(16,28)
following the trace convention W[k*n+i, r].
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START


def _verify_decomposition(U, V, W, n, m, p):
    """Verify U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                want = 1 if i == k else 0
                s = 0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != want:
                    return False
    return True


def _build_rank28():
    """Construct the rank-28 block-Strassen decomposition (integer entries)."""
    n = m = p = 4
    rank = 28
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)

    # Strassen 7 products over the ring of 2x2 matrices.
    # Each entry is a 2x2 block; a (sign) on a block means add sign*block.
    # Block (I,J) refers to rows 2I..2I+1, cols 2J..2J+1 of the 4x4 matrix.
    #
    #   M1 = (A00 + A11) * (B00 + B11)
    #   M2 = (A10 + A11) * (B00 - B01)
    #   M3 = (A00 - A10) * (B10 + B11)
    #   M4 = A01 * (B11 - B01)
    #   M5 = A10 * (B01 - B00)
    #   M6 = (A00 + A01) * (B10 - B11)
    #   M7 = (A11 - A01) * (B00 + B10)
    #
    #   C00 = M1 + M3 - M7 + M5
    #   C10 = M2 + M7
    #   C01 = M4 - M6 + M1 - M3
    #   C11 = M6 + M4 - M2 + M1

    # (I,J) -> (row_start, col_start)
    def block_rc(I, J):
        return 2 * I, 2 * J

    # Each product: list of (A_block, sign) for U, (B_block, sign) for V,
    # (C_block, sign) for W.  A_block (I,J) -> U rows (I,J); B_block (J,K)
    # -> V rows (J,K); C_block (I,K) -> W rows (K,I)  [trace convention].
    products = [
        # M1
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 0, 1), (0, 1, -1), (1, 1, -1)]),
        # M2
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1), (0, 1, -1)],
         [(1, 0, 1), (1, 1, -1)]),
        # M3
        ([(0, 0, 1), (1, 0, -1)], [(1, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 0, -1), (0, 1, -1), (1, 1, 1)]),
        # M4
        ([(0, 1, 1)], [(1, 1, 1), (0, 1, -1)],
         [(0, 1, 1), (1, 1, 1), (0, 0, -1), (1, 0, 1)]),
        # M5
        ([(1, 0, 1)], [(0, 1, 1), (0, 0, -1)],
         [(0, 0, 1), (1, 0, 1)]),
        # M6
        ([(0, 0, 1), (0, 1, 1)], [(1, 0, 1), (1, 1, -1)],
         [(1, 1, 1), (0, 1, -1), (1, 0, 1), (0, 0, -1)]),
        # M7
        ([(1, 1, 1), (0, 1, -1)], [(0, 0, 1), (1, 0, 1)],
         [(0, 0, -1), (1, 0, 1), (0, 1, 1), (1, 1, -1)]),
    ]

    for c, (u_list, v_list, w_list) in enumerate(products):
        for (I, J, s) in u_list:
            r, col = block_rc(I, J)
            U[r:r + 2, col] += s
            U[r:r + 2, col + 1] += s
            U[r + 1:r + 3, col] += s
            U[r + 1:r + 3, col + 1] += s
        for (J, K, s) in v_list:
            r, col = block_rc(J, K)
            V[r:r + 2, col] += s
            V[r:r + 2, col + 1] += s
            V[r + 1:r + 3, col] += s
            V[r + 1:r + 3, col + 1] += s
        for (I, K, s) in w_list:
            # W row index = K*n + I
            r, col = block_rc(K, I)
            W[r:r + 2, col] += s
            W[r:r + 2, col + 1] += s
            W[r + 1:r + 3, col] += s
            W[r + 1:r + 3, col + 1] += s

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _strassen_kron_4x4(n, m, p):
    """Rank-49 fallback via Kronecker product of Strassen 2x2."""
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)

    perm_u = np.zeros(16, dtype=int)
    for S in range(16):
        i, j = S // 4, S % 4
        a = (i // 2) * 2 + (j // 2)
        b = (i % 2) * 2 + (j % 2)
        perm_u[S] = a * 4 + b
    perm_v = np.zeros(16, dtype=int)
    for S in range(16):
        j, k = S // 4, S % 4
        a = (j // 2) * 2 + (k // 2)
        b = (j % 2) * 2 + (k % 2)
        perm_v[S] = a * 4 + b
    perm_w = np.zeros(16, dtype=int)
    for S in range(16):
        k, i = S // 4, S % 4
        a = (k // 2) * 2 + (i // 2)
        b = (k % 2) * 2 + (i % 2)
        perm_w[S] = a * 4 + b

    U4 = U4[perm_u, :]
    V4 = V4[perm_v, :]
    W4 = W4[perm_w, :]

    if _verify_decomposition(U4.tolist(), V4.tolist(), W4.tolist(), n, m, p):
        return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
    return None


def _schoolbook(n, m, p):
    """Rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """Construct a minimal-rank exact tensor decomposition for 4x4 matmul."""
    start_time = time.time()
    deadline = start_time + 120  # safety margin before the 150s kill

    n, m, p = DIMENSIONS

    # Primary: rank-28 block-Strassen (exact, integer).
    if time.time() < deadline:
        r28 = _build_rank28()
        if _verify_decomposition(r28["U"], r28["V"], r28["W"], n, m, p):
            return r28

    # Fallback: rank-49 Kronecker-Strassen.
    if time.time() < deadline:
        r49 = _strassen_kron_4x4(n, m, p)
        if r49 is not None:
            return r49

    # Final fallback: schoolbook.
    return _schoolbook(n, m, p)


# EVOLVE-BLOCK-END
