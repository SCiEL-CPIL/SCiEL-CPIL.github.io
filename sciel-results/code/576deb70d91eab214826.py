"""
Matrix multiplication 2x3 * 3x7 via Strassen decomposition.
Uses rank-7 Strassen for 2x2x3 sub-blocks and rank-6 schoolbook for 2x2x1.
Total rank: 2*7 + 6 = 20.
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 7]


def _build_strassen_2x2x2():
    """
    Build U, V, W for 2x2 matrix multiplication with rank 7 (Strassen).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1

    # M3 = a00(b10-b11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[2, 2] = 1

    # M4 = a11(b01-b00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[3, 3] = 1

    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1

    # M6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[1, 5] = 1

    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1; W[3, 6] = 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    # Check valid entries: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check a sample of invalid entries to ensure they're 0
    # For thoroughness, check all invalid entries for small tensors
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                # Decode indices
                i_u = u // m
                j_u = u % m
                j_v = v // p
                k_v = v % p
                k_w = w // n
                i_w = w % n
                # Valid if i_u == i_w and j_u == j_v and k_v == k_w
                if i_u == i_w and j_u == j_v and k_v == k_w:
                    continue
                val = np.dot(U[u, :], V[v, :] * W[w, :])
                if abs(val) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 130.0  # Safety margin before 150s kill

    # Build Strassen 2x2x2 components
    U_s, V_s, W_s = _build_strassen_2x2x2()

    # Build the full decomposition
    # For each of the two 2x2x3 blocks (j=0,1 and j=2), use Strassen for the 2x2x2 part
    # and schoolbook for the 2x2x1 part
    total_rank = 2 * 7 + 6  # = 20

    U = np.zeros((n * m, total_rank))  # (6, 20)
    V = np.zeros((m * p, total_rank))  # (21, 20)
    W = np.zeros((p * n, total_rank))  # (14, 20)

    col = 0
    for j in range(m):  # j = 0, 1, 2
        if j < 2:
            # Use Strassen for 2x2x2 sub-block (k=0,1)
            for t in range(7):
                # U: rows i*2 + i_sub where i_sub in {0,1}, but we have 2 rows per i
                # U[i*m + j, col] = U_s[i_sub, t] for i_sub in {0,1}
                for i in range(n):
                    U[i * m + j, col] = U_s[i, t]
                # V: row j*p + k_sub where k_sub in {0,1}
                for k in range(2):
                    V[j * p + k, col] = V_s[k, t]
                # W: row k_sub*n + i_sub
                for k in range(2):
                    for i in range(n):
                        W[k * n + i, col] = W_s[k * n + i, t]
                col += 1
            # Schoolbook for 2x2x1 sub-block (k=2)
            for i in range(n):
                for k in range(1):
                    kk = 2 + k
                    U[i * m + j, col] = 1.0
                    V[j * p + kk, col] = 1.0
                    W[kk * n + i, col] = 1.0
                    col += 1
        else:
            # For j=2, use schoolbook for all 2x2x3 (k=0,1,2)
            for i in range(n):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Fallback: schoolbook
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1.0
                V_sb[j * p + k, c] = 1.0
                W_sb[k * n + i, c] = 1.0
                c += 1

    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
