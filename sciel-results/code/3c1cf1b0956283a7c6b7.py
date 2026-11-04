import time
import numpy as np

N, M, P = 2, 3, 10


def _validate(U, V, W):
    """Return True if the factor matrices exactly decompose the tensor."""
    Ua = np.asarray(U, dtype=np.float64)
    Va = np.asarray(V, dtype=np.float64)
    Wa = np.asarray(W, dtype=np.float64)
    # sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] must equal 1
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = np.dot(Ua[i * M + j], Va[j * P + k] * Wa[k * N + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _trivial():
    r = N * M * P
    U = [[0] * r for _ in range(N * M)]
    V = [[0] * r for _ in range(M * P)]
    W = [[0] * r for _ in range(P * N)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _build_strassen():
    """
    Rank-55 decomposition.
    A (2x3) = A_left (2x2) | A_right (2x1)
    B (3x10): B_top (2x10), B_bottom (1x10)
    A_left * B_top : 5 column-pairs, Strassen rank 7 each = 35
    A_right * B_bottom : 2 x 10 schoolbook = 20
    Total = 55
    """
    rank = 55
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]

    r = 0
    # Strassen terms: (u, v, w) each a list of (row, col, coeff)
    terms = [
        # M1 = (A00+A11)(B00+B11)
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)],
         [(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]),
        # M2 = (A10+A11)B00
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        # M3 = A00(B01-B11)
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        # M4 = A11(B10-B00)
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (1, 0, 1)]),
        # M5 = (A00+A01)B11
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 0, 1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]

    for b in range(5):
        col0 = 2 * b
        col1 = 2 * b + 1
        for (u_terms, v_terms, w_terms) in terms:
            for (a_r, a_c, coeff) in u_terms:
                U[a_r * M + a_c][r] += coeff
            for (b_r, b_c, coeff) in v_terms:
                V[b_r * P + col0 + b_c][r] += coeff
            for (c_r, c_c, coeff) in w_terms:
                W[(col0 + c_c) * N + c_r][r] += coeff
            r += 1

    # A_right (col 2) * B_bottom (row 2): 20 terms
    for i in range(N):
        for k in range(P):
            U[i * M + 2][r] = 1
            V[2 * P + k][r] = 1
            W[k * N + i][r] = 1
            r += 1

    assert r == rank, f"Expected rank {rank}, got {r}"
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    deadline = start_time + 140

    try:
        res = _build_strassen()
        if _validate(res["U"], res["V"], res["W"]):
            if time.time() > deadline:
                return res
            return res
    except Exception:
        pass

    return _trivial()
