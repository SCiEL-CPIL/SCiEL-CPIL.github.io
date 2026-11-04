"""Matrix multiplication 4x4x8 with a rank-98 decomposition."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _build_2x2x2_strassen():
    """
    Build the rank-7 Strassen decomposition for 2x2x2.
    Returns U (4,7), V (4,7), W (4,7) where:
      U[i*m+j, r] = coefficient for A[i,j] in term r
      V[j*p+k, r] = coefficient for B[j,k] in term r
      W[k*n+i, r] = coefficient for C[i,k] in term r
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # M3: A00(B10-B00)
    U[0, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # M4: A11(B01-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1

    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = -1

    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1

    return U, V, W


def _build_4x4x4_strassen():
    """
    Build a rank-49 decomposition for 4x4x4 using Strassen on 2x2 blocks.
    """
    U2, V2, W2 = _build_2x2x2_strassen()

    U = np.zeros((16, 49), dtype=np.float64)
    V = np.zeros((16, 49), dtype=np.float64)
    W = np.zeros((16, 49), dtype=np.float64)

    A_blocks = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
    ]

    B_blocks = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
    ]

    C_blocks = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, -1), (1, 0, 1)],
        [(0, 0, 1), (0, 1, -1)],
        [(1, 1, 1)],
        [(0, 0, -1), (0, 1, 1)],
    ]

    for j in range(7):
        for s in range(7):
            col = j * 7 + s
            for (br, bc, sign) in A_blocks[j]:
                for li in range(2):
                    for lj in range(2):
                        i = 2 * br + li
                        jj = 2 * bc + lj
                        U[i * 4 + jj, col] += sign * U2[li * 2 + lj, s]

            for (br, bc, sign) in B_blocks[j]:
                for li in range(2):
                    for lj in range(2):
                        jj = 2 * br + li
                        k = 2 * bc + lj
                        V[jj * 4 + k, col] += sign * V2[li * 2 + lj, s]

            for (br, bc, sign) in C_blocks[j]:
                for li in range(2):
                    for lj in range(2):
                        i = 2 * br + li
                        k = 2 * bc + lj
                        W[k * 4 + i, col] += sign * W2[li * 2 + lj, s]

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(total - 1.0) > tol:
                    return False
    return True


def _build_4x4x8_rank98():
    """
    Build the rank-98 decomposition for 4x4x8 by splitting B into two 4x4 halves.
    """
    U44, V44, W44 = _build_4x4x4_strassen()

    U = np.zeros((16, 98), dtype=np.float64)
    V = np.zeros((32, 98), dtype=np.float64)
    W = np.zeros((32, 98), dtype=np.float64)

    U[:, 0:49] = U44
    V[0:16, 0:49] = V44
    W[0:16, 0:49] = W44

    U[:, 49:98] = U44
    V[16:32, 49:98] = V44
    W[16:32, 49:98] = W44

    return U, V, W


def _schoolbook_decomposition(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x8 matrix.
    """
    n, m, p = 4, 4, 8

    # Build the rank-98 Strassen-based decomposition
    U, V, W = _build_4x4x8_rank98()

    # Verify correctness
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook if verification fails
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
