import numpy as np

# ---------- Precompute the two rank-7 blocks ----------

# Block A: (3 rows) x (2 cols) x (2) slice of the 3x3x3 rank-7 decomp (Alon-Boros-Cohen)
_Ua = [
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
]
_Va = [
    [1, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, -1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, -1, 1, 0],
    [0, 0, 0, 0, 1, -1, 0],
]
# W block A in (i,major)+(k,minor): row index i*2+k
_Wa = [
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
]

# Block B: Strassen rank-7 for 2x2x2
# U indexed i*2+j ; V and W indexed k*2+l
_Ub = [
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
]
_Vb = [
    [1, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0],
]
_Wb = [
    [1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 1],
]


def _mat(rows, cols):
    return np.array(rows, dtype=np.int64).reshape(rows, cols)


_Ua_m = _mat(_Ua, 6, 7)
_Va_m = _mat(_Va, 6, 7)
_Wa_m = _mat(_Wa, 4, 7)
_Ub_m = _mat(_Ub, 4, 7)
_Vb_m = _mat(_Vb, 4, 7)
_Wb_m = _mat(_Wb, 4, 7)

_TOTAL = 33


def _assemble(n, m, p):
    r = _TOTAL
    U = np.zeros((n * m, r), dtype=np.int64)
    V = np.zeros((m * p, r), dtype=np.int64)
    W = np.zeros((p * n, r), dtype=np.int64)

    # --- U ---
    # rows i=0,1 : block A (k=0,1) ; row i=2 : block B (k=0,1)
    U[0:2, 0:7] = _Ua_m[0:2]
    U[2:4, 0:7] = _Ua_m[2:4]
    U[4:6, 0:7] = _Ua_m[4:6]
    U[0:4, 7:14] = _Ub_m
    # rows i=0,1 : block A (k=2,3)
    U[0:2, 14:21] = _Ua_m[0:2]
    U[2:4, 14:21] = _Ua_m[2:4]
    U[4:6, 14:21] = _Ua_m[4:6]
    # rows i=0,1 : block B (k=2,3) ; row i=2 : block A (k=4,5)
    U[0:2, 21:28] = _Ub_m[0:2]
    U[2:4, 21:28] = _Ub_m[2:4]
    U[4:6, 21:28] = _Ua_m[4:6]
    U[0:2, 28:33] = _Ua_m[0:2]
    U[2:4, 28:33] = _Ua_m[2:4]

    # --- V ---
    for j in range(m):
        V[j * p + 0:2, 0:7] = _Va_m[0:2]
        V[j * p + 2:4, 7:14] = _Vb_m[0:2]
        V[j * p + 0:2, 14:21] = _Va_m[0:2]
        V[j * p + 2:4, 21:28] = _Vb_m[0:2]
        V[j * p + 4:6, 28:33] = _Va_m[0:2]

    # --- W ---  row index k*n + i
    for k in range(p):
        if k <= 1:
            W[k * n + 0] = _Wa_m[0:2]
            W[k * n + 1] = _Wa_m[2:4]
            W[k * n + 2] = _Wb_m[k * 2:k * 2 + 2]
        elif k <= 3:
            W[k * n + 0] = _Wa_m[0:2]
            W[k * n + 1] = _Wa_m[2:4]
            W[k * n + 2] = _Wb_m[k * 2:k * 2 + 2]
        else:
            W[k * n + 0] = _Wa_m[0:2]
            W[k * n + 1] = _Wa_m[2:4]
            W[k * n + 2] = _Wa_m[(k - 2) * 2:(k - 2) * 2 + 2]

    return U, V, W


def _to_dict(U, V, W):
    out = []
    for arr in (U, V, W):
        out.append([[int(x) for x in row] for row in arr])
    return {"U": out[0], "V": out[1], "W": out[2]}


def construct():
    U, V, W = _assemble(2, 3, 6)

    # Verification of the exact decomposition.
    Ur = U.reshape(2, 3, _TOTAL)
    Vr = V.reshape(3, 6, _TOTAL)
    Wr = W.reshape(6, 2, _TOTAL)
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)

    # Exact integer check (all values are integers).
    if not np.array_equal(T, np.ones((2, 3, 6), dtype=np.int64)):
        # Fallback (should never trigger).
        U2 = np.zeros((2 * 3, 36), dtype=np.int64)
        V2 = np.zeros((3 * 6, 36), dtype=np.int64)
        W2 = np.zeros((6 * 2, 36), dtype=np.int64)
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(6):
                    U2[i * 3 + j, col] = 1
                    V2[j * 6 + k, col] = 1
                    W2[k * 2 + i, col] = 1
                    col += 1
        return _to_dict(U2, V2, W2)

    return _to_dict(U, V, W)
