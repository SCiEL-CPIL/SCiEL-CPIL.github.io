import time
import numpy as np

DIMENSIONS = [4, 4, 9]


def _build_strassen_2x2():
    U = np.zeros((4, 7), dtype=np.int32)
    V = np.zeros((4, 7), dtype=np.int32)
    W = np.zeros((4, 7), dtype=np.int32)

    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_4x4x4_strassen():
    U2, V2, W2 = _build_strassen_2x2()

    U = np.zeros((16, 49), dtype=np.int32)
    V = np.zeros((16, 49), dtype=np.int32)
    W = np.zeros((16, 49), dtype=np.int32)

    big_mults = [
        {'A': [((0, 0), 1), ((1, 1), 1)], 'B': [((0, 0), 1), ((1, 1), 1)], 'C': [((0, 0), 1), ((1, 1), 1)]},
        {'A': [((1, 0), 1), ((1, 1), 1)], 'B': [((0, 0), 1)], 'C': [((0, 1), 1), ((1, 1), -1)]},
        {'A': [((0, 0), 1)], 'B': [((0, 1), 1), ((1, 1), -1)], 'C': [((1, 0), 1), ((1, 1), 1)]},
        {'A': [((1, 1), 1)], 'B': [((1, 0), 1), ((0, 0), -1)], 'C': [((0, 0), 1), ((0, 1), 1)]},
        {'A': [((0, 0), 1), ((0, 1), 1)], 'B': [((1, 1), 1)], 'C': [((0, 0), -1), ((1, 0), 1)]},
        {'A': [((1, 0), 1), ((0, 0), -1)], 'B': [((0, 0), 1), ((0, 1), 1)], 'C': [((1, 1), 1)]},
        {'A': [((0, 1), 1), ((1, 1), -1)], 'B': [((1, 0), 1), ((1, 1), 1)], 'C': [((0, 0), 1)]},
    ]

    for big_idx, bm in enumerate(big_mults):
        for r2 in range(7):
            col = big_idx * 7 + r2

            for (bi, bj), sign_a in bm['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        u_val = U2[i2 * 2 + k2, r2]
                        if u_val != 0:
                            I = 2 * bi + i2
                            J = 2 * bj + k2
                            U[I * 4 + J, col] += sign_a * u_val

            for (bk, bj), sign_b in bm['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        v_val = V2[k2 * 2 + j2, r2]
                        if v_val != 0:
                            J = 2 * bk + k2
                            K = 2 * bj + j2
                            V[J * 4 + K, col] += sign_b * v_val

            for (bi, bj), sign_c in bm['C']:
                for i2 in range(2):
                    for j2 in range(2):
                        w_val = W2[i2 * 2 + j2, r2]
                        if w_val != 0:
                            I = 2 * bi + i2
                            J = 2 * bj + j2
                            W[I * 4 + J, col] += sign_c * w_val

    return U, V, W


def _schoolbook_fallback():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    return {"U": U, "V": V, "W": W}


def _validate_numpy(U, V, W, n, m, p):
    U_np = np.asarray(U, dtype=np.int32)
    V_np = np.asarray(V, dtype=np.int32)
    W_np = np.asarray(W, dtype=np.int32)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i]
                if np.sum(prod) != 1:
                    return False
    return True


def _build_4x4x9_strassen():
    U444, V444, W444 = _build_4x4x4_strassen()

    n, m, p = 4, 4, 9
    total_rank = 49 + 49 + 16
    U = np.zeros((n * m, total_rank), dtype=np.int32)
    V = np.zeros((m * p, total_rank), dtype=np.int32)
    W = np.zeros((p * n, total_rank), dtype=np.int32)

    # Block 1: output columns 0..3
    U[:, 0:49] = U444
    for k in range(m):
        for j in range(4):
            V[k * p + j, 0:49] = V444[k * 4 + j, :]
    for k in range(4):
        for i in range(n):
            W[k * n + i, 0:49] = W444[k * 4 + i, :]

    # Block 2: output columns 4..7
    off = 49
    U[:, off:off + 49] = U444
    for k in range(m):
        for j in range(4):
            V[k * p + (j + 4), off:off + 49] = V444[k * 4 + j, :]
    for k in range(4):
        for i in range(n):
            W[(k + 4) * n + i, off:off + 49] = W444[k * 4 + i, :]

    # Block 3: output column 8, schoolbook rank 16
    off3 = 98
    for i in range(n):
        for j in range(m):
            idx = i * m + j
            U[i * m + j, off3 + idx] = 1
            V[j * p + 8, off3 + idx] = 1
            W[8 * n + i, off3 + idx] = 1

    return U, V, W


def _try_winograd_4x4x9():
    """
    Attempt to construct a lower-rank decomposition using Winograd-style
    frequency-domain factorization on the 4x4x9 tensor.
    
    Idea: For the 4x4x9 case, we can try to use a block structure where
    we treat the 9 output columns in a way that shares computation.
    
    One approach: use the Strassen 4x4x4 result but try to overlap
    the two 4x4x4 blocks by sharing some rank terms.
    
    For 4x4 x 4x9, we can think of it as two 4x4 x 4x4 multiplications
    sharing the same A matrix. The two blocks both use U444 for A,
    but different V and W for B and C.
    
    The total rank is 49 + 49 + 16 = 114.
    
    To reduce rank, we need to find shared rank terms between blocks.
    This is equivalent to finding a lower-rank decomposition of the
    4x4x9 tensor directly.
    
    Let's try a direct approach using iterative rank reduction.
    """
    n, m, p = 4, 4, 9
    
    # Start with the Strassen-based decomposition
    U, V, W = _build_4x4x9_strassen()
    rank = U.shape[1]
    
    # Try to reduce rank by merging columns that are linearly dependent
    # or by finding a lower-rank equivalent
    
    # For now, return the same result - this is a placeholder for
    # future optimization attempts
    return U, V, W, rank


def construct():
    start = time.time()
    deadline = start + 110.0

    U, V, W = _build_4x4x9_strassen()

    if not _validate_numpy(U, V, W, 4, 4, 9):
        return _schoolbook_fallback()

    # Check time
    if time.time() > deadline:
        return _schoolbook_fallback()

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return result
