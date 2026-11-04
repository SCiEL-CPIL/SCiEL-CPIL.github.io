"""
Matrix multiplication 2x4 by 4x4, exact tensor decomposition.
Uses Strassen's algorithm on four 2x2 sub-blocks to achieve rank 28 (vs schoolbook 32).
The code is restructured from the prior baseline to use flat index arithmetic,
vectorised verification, and an explicit deadline guard, while preserving the
same mathematical construction.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()


def construct():
    best_result = None
    deadline = START_TIME + max(0.0, TIME_LIMIT - 10.0)
    try:
        best_result = _construct_strassen_28()
        if best_result is not None and _verify_decomposition(best_result):
            return best_result
    except Exception:
        best_result = None
    if time.time() > deadline:
        return _construct_schoolbook_32()
    try:
        best_result = _construct_schoolbook_32()
        if _verify_decomposition(best_result):
            return best_result
    except Exception:
        pass
    return _construct_schoolbook_32()


def _verify_decomposition(result):
    try:
        n, m, p = DIMENSIONS
        U = np.asarray(result["U"], dtype=float)
        V = np.asarray(result["V"], dtype=float)
        W = np.asarray(result["W"], dtype=float)

        rng = np.random.default_rng(12345)
        for _ in range(3):
            A = rng.random((n, m))
            B = rng.random((m, p))
            C_expected = A @ B

            a_flat = A.reshape(-1)
            b_flat = B.reshape(-1)
            A_t = a_flat @ U
            B_t = b_flat @ V
            C_flat = (A_t * B_t) @ W.T
            C_computed = C_flat.reshape((n, p))

            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    except Exception:
        return False


def _construct_schoolbook_32():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 32
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


def _construct_strassen_28():
    """
    2x4 * 4x4 via four independent 2x2 Strassen products.

    A is 2x4, B is 4x4, C is 2x4.
    Partition A into two 2x2 blocks horizontally:
        A = [A1 | A2]
    Partition B into four 2x2 blocks:
        B = [[B11, B12],
             [B21, B22]]
    Then C = [C1 | C2] where
        C1 = A1*B11 + A2*B21
        C2 = A1*B12 + A2*B22

    Each 2x2 product uses the standard Strassen 7-term decomposition.
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen terms for a 2x2 product X*Y.
    # Local flat index convention: x00=0, x01=1, x10=2, x11=3.
    # Each term is (u_coeffs, v_coeffs, w_coeffs) with flat local indices.
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # P1
        ({2: 1, 3: 1}, {0: 1},       {2: 1, 3: -1}),   # P2
        ({0: 1},       {1: 1, 3: -1},{1: 1, 3: 1}),    # P3
        ({3: 1},       {0: -1, 2: 1},{0: 1, 2: 1}),    # P4
        ({0: 1, 1: 1}, {3: 1},       {0: -1, 1: 1}),   # P5
        ({0: -1, 2: 1},{0: 1, 1: 1}, {3: 1}),          # P6
        ({1: 1, 3: -1},{2: 1, 3: 1}, {0: 1}),          # P7
    ]

    # For each sub-product, define flat-index maps from local 2x2 index to global flat index.
    # A local index l -> (r, c) with r = l // 2, c = l % 2.
    # a_map: (r, c) -> global A index
    # b_map: (r, c) -> global B index
    # c_map: (r, c) -> global C index (trace convention W[k*n+i])

    sub_products = [
        # 1. A1 (cols 0-1) * B11 (rows 0-1, cols 0-1) -> C1 (cols 0-1)
        {
            'a': lambda r, c: r * 4 + c,
            'b': lambda r, c: r * 4 + c,
            'c': lambda r, c: c * 2 + r,
        },
        # 2. A2 (cols 2-3) * B21 (rows 2-3, cols 0-1) -> C1 (cols 0-1)
        {
            'a': lambda r, c: r * 4 + c + 2,
            'b': lambda r, c: r * 4 + c + 8,
            'c': lambda r, c: c * 2 + r,
        },
        # 3. A1 (cols 0-1) * B12 (rows 0-1, cols 2-3) -> C2 (cols 2-3)
        {
            'a': lambda r, c: r * 4 + c,
            'b': lambda r, c: r * 4 + c + 2,
            'c': lambda r, c: (c + 2) * 2 + r,
        },
        # 4. A2 (cols 2-3) * B22 (rows 2-3, cols 2-3) -> C2 (cols 2-3)
        {
            'a': lambda r, c: r * 4 + c + 2,
            'b': lambda r, c: r * 4 + c + 10,
            'c': lambda r, c: (c + 2) * 2 + r,
        },
    ]

    col = 0
    for sp in sub_products:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc, coeff in u_coeffs.items():
                r = loc // 2
                c = loc % 2
                U[sp['a'](r, c)][col] += coeff
            for loc, coeff in v_coeffs.items():
                r = loc // 2
                c = loc % 2
                V[sp['b'](r, c)][col] += coeff
            for loc, coeff in w_coeffs.items():
                r = loc // 2
                c = loc % 2
                W[sp['c'](r, c)][col] += coeff
            col += 1

    if col != 28:
        raise ValueError(f"Expected col=28, got {col}")

    return {"U": U, "V": V, "W": W}
