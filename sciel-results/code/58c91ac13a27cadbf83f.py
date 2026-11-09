import time
import numpy as np

_DIM = (6, 6, 6)
_DEADLINE = 145.0  # seconds, with 5s safety margin


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    A = np.random.randint(0, 3, (n, m)).astype(float)
    B = np.random.randint(0, 3, (m, p)).astype(float)
    C_expected = A @ B

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for r in range(rank):
                u_val = 0.0
                for j in range(m):
                    u_val += A[i, j] * U_arr[i * m + j, r]
                v_val = 0.0
                for j in range(m):
                    v_val += B[j, k] * V_arr[j * p + k, r]
                w_val = W_arr[k * n + i, r]
                s += u_val * v_val * w_val
            C_computed[i, k] = s

    max_err = np.max(np.abs(C_computed - C_expected))
    return max_err


def _laderman_2x2_terms():
    """
    Laderman's 6-term decomposition for 2x2 matrix multiplication.
    Each term is a dict with 'U', 'V', 'W' as lists of (row, col, coeff).
    U indexed by (i,j) for A[i,j], V by (j,k) for B[j,k], W by (i,k) for C[i,k].
    """
    terms = [
        # M1 = A00 * B00
        {'U': [(0, 0, 1)], 'V': [(0, 0, 1)], 'W': [(0, 0, 1)]},
        # M2 = (A00 + A01) * B10
        {'U': [(0, 0, 1), (0, 1, 1)], 'V': [(1, 0, 1)], 'W': [(0, 1, 1), (1, 1, 1)]},
        # M3 = (A10 - A00) * B01
        {'U': [(1, 0, 1), (0, 0, -1)], 'V': [(0, 1, 1)], 'W': [(1, 0, 1), (1, 1, 1)]},
        # M4 = A11 * B11
        {'U': [(1, 1, 1)], 'V': [(1, 1, 1)], 'W': [(1, 1, 1)]},
        # M5 = (A01 - A11) * (B10 + B11)
        {'U': [(0, 1, 1), (1, 1, -1)], 'V': [(1, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1)]},
        # M6 = (A10 + A11) * (B00 - B01)
        {'U': [(1, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1), (0, 1, -1)], 'W': [(0, 0, 1), (1, 0, 1)]},
    ]
    return terms


def _schoolbook_3x3():
    """Schoolbook 27-term decomposition for 3x3 matrix multiplication."""
    rank = 27
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]

    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                U[i * 3 + j][r] = 1
                V[j * 3 + k][r] = 1
                W[k * 3 + i][r] = 1

    return U, V, W


def _build_laderman_6x6_3x3_blocks():
    """
    2x2 blocks of size 3, Laderman (rank 6) on 2x2 super-matrix.
    Each 3x3 multiplication uses schoolbook (27 terms).
    Total rank: 6 * 27 = 162.
    """
    n, m, p = 6, 6, 6
    B = 3
    N_b = n // B  # 2
    rank = 6 * 27  # 162

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    laderman_terms = _laderman_2x2_terms()
    schoolbook_3x3_U, schoolbook_3x3_V, schoolbook_3x3_W = _schoolbook_3x3()

    for t in range(6):
        term = laderman_terms[t]
        for lr, lc, coeff in term['U']:
            for di in range(3):
                for dj in range(3):
                    u_row = (lr * B + di) * m + (lc * B + dj)
                    u_idx_local = (lr * 3 + di) * 3 + (lc * 3 + dj)
                    for r_local in range(27):
                        if schoolbook_3x3_U[u_idx_local][r_local] != 0:
                            col_idx = t * 27 + r_local
                            U[u_row][col_idx] += coeff * schoolbook_3x3_U[u_idx_local][r_local]

        for lr, lc, coeff in term['V']:
            for di in range(3):
                for dj in range(3):
                    v_row = (lr * B + di) * p + (lc * B + dj)
                    v_idx_local = (lr * 3 + di) * 3 + (lc * 3 + dj)
                    for r_local in range(27):
                        if schoolbook_3x3_V[v_idx_local][r_local] != 0:
                            col_idx = t * 27 + r_local
                            V[v_row][col_idx] += coeff * schoolbook_3x3_V[v_idx_local][r_local]

        for lr, lc, coeff in term['W']:
            for di in range(3):
                for dj in range(3):
                    w_row = (lc * B + dj) * n + (lr * B + di)
                    w_idx_local = (lc * 3 + dj) * 3 + (lr * 3 + di)
                    for r_local in range(27):
                        if schoolbook_3x3_W[w_idx_local][r_local] != 0:
                            col_idx = t * 27 + r_local
                            W[w_row][col_idx] += coeff * schoolbook_3x3_W[w_idx_local][r_local]

    return U, V, W


def _strassen_2x2_terms():
    """Strassen's 7 terms for 2x2 matrix multiplication."""
    terms = [
        {'U': [(0, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1), (1, 1, 1)]},
        {'U': [(1, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1)], 'W': [(1, 0, 1), (1, 1, -1)]},
        {'U': [(0, 0, 1)], 'V': [(0, 1, 1), (1, 1, -1)], 'W': [(0, 1, 1), (1, 1, 1)]},
        {'U': [(1, 1, 1)], 'V': [(1, 0, 1), (0, 0, -1)], 'W': [(0, 0, 1), (1, 0, 1)]},
        {'U': [(0, 0, 1), (0, 1, 1)], 'V': [(1, 1, 1)], 'W': [(0, 0, -1), (0, 1, 1)]},
        {'U': [(1, 0, 1), (0, 0, -1)], 'V': [(0, 0, 1), (0, 1, 1)], 'W': [(1, 1, 1)]},
        {'U': [(0, 1, 1), (1, 1, -1)], 'V': [(1, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1)]},
    ]
    return terms


def _build_strassen_6x6_2x2_blocks():
    """3x3 blocks of size 2, Strassen on each 2x2 block. Rank 189."""
    n, m, p = 6, 6, 6
    B = 2
    N_b = n // B  # 3
    rank = N_b * N_b * N_b * 7  # 189

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_terms = _strassen_2x2_terms()

    for i in range(N_b):
        for k in range(N_b):
            for l in range(N_b):
                for t in range(7):
                    col_idx = (i * N_b + k) * (N_b * 7) + l * 7 + t
                    term = strassen_terms[t]

                    for lr, lc, coeff in term['U']:
                        u_row = (i * B + lr) * m + (k * B + lc)
                        U[u_row][col_idx] += coeff

                    for lr, lc, coeff in term['V']:
                        v_row = (k * B + lr) * p + (l * B + lc)
                        V[v_row][col_idx] += coeff

                    for lr, lc, coeff in term['W']:
                        w_row = (l * B + lc) * n + (i * B + lr)
                        W[w_row][col_idx] += coeff

    return U, V, W


def construct():
    n, m, p = _DIM
    start_time = time.time()

    best_result = None
    best_rank = float('inf')

    # Approach 1: Laderman 2x2 super-matrix with 3x3 schoolbook blocks (rank 162)
    try:
        U, V, W = _build_laderman_6x6_3x3_blocks()
        rank1 = len(U[0])
        err = _verify_decomposition(U, V, W, n, m, p)
        if err < 1e-6:
            if rank1 < best_rank:
                best_rank = rank1
                best_result = {"U": U, "V": V, "W": W}
    except Exception:
        pass

    if time.time() - start_time > _DEADLINE - 20:
        if best_result is not None:
            return best_result
        return _build_strassen_6x6_2x2_blocks()

    # Approach 2: Strassen 3x3 blocks of size 2 (rank 189)
    try:
        U, V, W = _build_strassen_6x6_2x2_blocks()
        rank2 = len(U[0])
        err = _verify_decomposition(U, V, W, n, m, p)
        if err < 1e-6:
            if rank2 < best_rank:
                best_rank = rank2
                best_result = {"U": U, "V": V, "W": W}
    except Exception:
        pass

    if best_result is not None:
        return best_result

    # Fallback: pure schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return {"U": U, "V": V, "W": W}
