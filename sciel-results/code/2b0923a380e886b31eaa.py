"""
Matrix multiplication 2x4 by 4x4, exact tensor decomposition.
Rank-28 decomposition via Strassen's algorithm on four 2x2 sub-blocks.
The program is deterministic and fast; no time-budget search is required.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()


def check_time_and_return(best_result):
    if best_result is not None and (time.time() - START_TIME) > TIME_LIMIT - 10.0:
        return best_result
    return None


def construct():
    best_result = None
    try:
        best_result = _construct_strassen_28()
        if best_result is not None:
            if _verify_decomposition(best_result):
                return best_result
    except Exception:
        pass
    if best_result is None:
        best_result = _construct_schoolbook_32()
    return best_result


def _verify_decomposition(result):
    """Verify that the decomposition correctly computes matrix multiplication."""
    try:
        n, m, p = DIMENSIONS
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)

        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B

        A_flat = A.flatten()
        A_t = A_flat @ U
        B_flat = B.flatten()
        B_t = B_flat @ V
        C_flat = (A_t * B_t) @ W.T
        C_computed = C_flat.reshape((n, p))

        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
        return True
    except Exception:
        return False


def _construct_schoolbook_32():
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


def _construct_strassen_28():
    """
    2x4 * 4x4 using four 2x2 Strassen blocks.
    Rank 28 (4 blocks x 7 Strassen multiplications).
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    sub_products = [
        {
            'a_map': lambda r, c: r * 4 + c,
            'b_map': lambda r, c: r * 4 + c,
            'c_map': lambda r, c: c * 2 + r,
        },
        {
            'a_map': lambda r, c: r * 4 + (c + 2),
            'b_map': lambda r, c: (r + 2) * 4 + c,
            'c_map': lambda r, c: c * 2 + r,
        },
        {
            'a_map': lambda r, c: r * 4 + c,
            'b_map': lambda r, c: r * 4 + (c + 2),
            'c_map': lambda r, c: (c + 2) * 2 + r,
        },
        {
            'a_map': lambda r, c: r * 4 + (c + 2),
            'b_map': lambda r, c: (r + 2) * 4 + (c + 2),
            'c_map': lambda r, c: (c + 2) * 2 + r,
        },
    ]

    col = 0
    for sp in sub_products:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_u, coeff in u_coeffs.items():
                r, c = loc_u // 2, loc_u % 2
                global_u = sp['a_map'](r, c)
                U[global_u][col] += coeff
            for loc_v, coeff in v_coeffs.items():
                r, c = loc_v // 2, loc_v % 2
                global_v = sp['b_map'](r, c)
                V[global_v][col] += coeff
            for loc_w, coeff in w_coeffs.items():
                r, c = loc_w // 2, loc_w % 2
                global_w = sp['c_map'](r, c)
                W[global_w][col] += coeff
            col += 1

    assert col == 28, f"Expected col=28, got {col}"

    return {"U": U, "V": V, "W": W}
