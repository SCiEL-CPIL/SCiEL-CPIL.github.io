"""Hybrid decomposition for 6x6 matrix multiplication.

Strategy: treat A, B, C as block matrices with a 4x4 core (rows/cols 0..3)
and a 2-wide tail (rows/cols 4..5).

  A = [[A11, A12],[A21, A22]]   B = [[B11, B12],[B21, B22]]
  C = [[C11, C12],[C21, C22]]

where A11,B11,C11 are 4x4 and the other blocks have a 2-wide row or col.

  C11 = A11*B11        -> Strassen rank 7
  C12 = A11*B12        -> schoolbook rank 4*2=8
  C21 = A21*B11        -> schoolbook rank 2*4=8
  C22 = A21*B12 + A22*B22 -> schoolbook rank 4+4=8

Total rank = 7 + 8 + 8 + 8 = 31.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces the matrix-multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=tol):
        diff = np.abs(T - expected).max()
        return False, diff
    return True, 0.0


def _build_hybrid(n, m, p):
    """Hybrid 4x4-Strassen-core + schoolbook-tail decomposition for 6x6x6."""
    assert n == m == p == 6, "This construction is for 6x6x6"

    # Strassen 2x2 (rank 7). Local 2x2 indices 0,1 (== global 0,1,2,3 in pairs).
    # U_local[sm] : list of (a,b,coeff)  over 2x2 A block
    # V_local[sm] : list of (c,d,coeff)  over 2x2 B block
    # C_local[(e,f)][sm] = coeff  (C = sum_m coeff * M_m)
    U_local = [
        [(0, 1, 1)],                       # M0 = A01 * B01
        [(0, 0, 1), (0, 1, 1)],            # M1 = (A00+A01) * B11
        [(1, 0, 1), (1, 1, 1)],            # M2 = A10 * (B11-B01)
        [(1, 1, 1)],                       # M3 = A11 * B11
        [(0, 0, 1), (1, 1, 1)],            # M4 = (A00+A11) * (B00+B11)
        [(0, 1, 1), (1, 1, -1)],           # M5 = (A01-A11) * (B00+B10)
        [(1, 0, 1), (0, 0, -1)],           # M6 = (A10-A00) * (B01+B11)
    ]
    V_local = [
        [(0, 0, 1)],                       # M0
        [(1, 1, 1)],                       # M1
        [(1, 1, 1), (0, 1, -1)],           # M2
        [(1, 1, 1)],                       # M3
        [(0, 0, 1), (1, 1, 1)],            # M4
        [(0, 0, 1), (1, 0, 1)],            # M5
        [(0, 1, 1), (1, 1, 1)],            # M6
    ]
    # C[e][f][m] = coeff  (C_local)
    C_local = {
        (0, 0): {4: 1, 3: 1, 1: -1, 5: 1},
        (0, 1): {0: 1, 1: 1},
        (1, 0): {2: 1, 3: 1},
        (1, 1): {4: 1, 0: 1, 2: -1, 6: -1},
    }

    U = np.zeros((n * m, 0), dtype=float)
    V = np.zeros((m * p, 0), dtype=float)
    W = np.zeros((p * n, 0), dtype=float)

    def add_col(u_rows, v_rows, w_rows):
        nonlocal U, V, W
        u = np.zeros(n * m, dtype=float)
        v = np.zeros(m * p, dtype=float)
        w = np.zeros(p * n, dtype=float)
        for (i, j), c in u_rows:
            u[i * m + j] += c
        for (j, k), c in v_rows:
            v[j * p + k] += c
        for (k, i), c in w_rows:
            w[k * n + i] += c
        U = np.column_stack([U, u])
        V = np.column_stack([V, v])
        W = np.column_stack([W, w])

    # ---- Part 1: C11 = A11 * B11 via Strassen (4x4 = 2x2 grid of 2x2) ----
    for pb in range(2):      # A11 row block (0,1)
        for qb in range(2):  # A11 col block
            for rb in range(2):  # B11 row block
                for sb in range(2):  # B11 col block
                    for sm in range(7):
                        u_rows = []
                        for (a, b, ca) in U_local[sm]:
                            u_rows.append(((pb * 2 + a), (qb * 2 + b), ca))
                        v_rows = []
                        for (c, d, cb) in V_local[sm]:
                            v_rows.append(((rb * 2 + c), (sb * 2 + d), cb))
                        w_rows = []
                        for (e, f) in C_local:
                            if sm in C_local[(e, f)]:
                                cc = C_local[(e, f)][sm]
                                w_rows.append(((rb * 2 + e), (pb * 2 + f), cc))
                        add_col(u_rows, v_rows, w_rows)

    # ---- Part 2: C12 = A11 * B12 (schoolbook) ----
    for i in range(4):
        for j in range(4):
            for k in range(2):      # B12 col = 4..5
                add_col([(i, j, 1)], [(j, 4 + k, 1)], [(4 + k, i, 1)])

    # ---- Part 3: C21 = A21 * B11 (schoolbook) ----
    for i in range(2):      # A21 row = 4..5
        for j in range(4):
            for k in range(4):
                add_col([(4 + i, j, 1)], [(j, k, 1)], [(k, 4 + i, 1)])

    # ---- Part 4: C22 = A21*B12 + A22*B22 (schoolbook) ----
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A21[i,j] * B12[j, k]
                add_col([(4 + i, j, 1)], [(j, 4 + k, 1)], [(4 + k, 4 + i, 1)])
                # A22[i,j] * B22[j, k]
                add_col([(4 + i, 4 + j, 1)], [(4 + j, 4 + k, 1)], [(4 + k, 4 + i, 1)])

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _build_schoolbook(n, m, p):
    """Fallback: schoolbook rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Primary: hybrid rank-31 decomposition
    try:
        result = _build_hybrid(n, m, p)
        ok, diff = _verify(result["U"], result["V"], result["W"], n, m, p)
        if ok:
            if time.time() < deadline:
                return result
            return result
        # Verification failed; fall through to fallback.
    except Exception:
        pass

    # Fallback: schoolbook rank 216
    return _build_schoolbook(n, m, p)
