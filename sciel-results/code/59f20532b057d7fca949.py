import time
import numpy as np

DIMENSIONS = [4, 4, 9]

_STRASSEN_222 = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
    ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
]


def _add_strassen_block(U, V, W, col, a_base, b_base, c_base):
    for (u, v, w) in _STRASSEN_222:
        for t in range(4):
            if u[t]:
                U[a_base + t][col] += u[t]
            if v[t]:
                V[b_base + t][col] += v[t]
            if w[t]:
                W[c_base + t][col] += w[t]
        col += 1
    return col


def _add_444(U, V, W, col, a_base, b_base, c_base):
    U_sums = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 1],
    ]
    V_sums = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    W_sums = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    for t in range(7):
        u, v, w = U_sums[t], V_sums[t], W_sums[t]
        for s in range(4):
            if u[s]:
                U[a_base + s][col] += u[s]
            if v[s]:
                V[b_base + s][col] += v[s]
            if w[s]:
                W[c_base + s][col] += w[s]
        col += 1
    return col


def _schoolbook_block(U, V, W, col, a_base, b_base, c_base, ar, ac, bc):
    for i in range(ar):
        for j in range(ac):
            for k in range(bc):
                U[a_base + i * ac + j][col] += 1
                V[b_base + j * bc + k][col] += 1
                W[c_base + i * bc + k][col] += 1
                col += 1
    return col


def _verify(U, V, W, n, m, p):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for r in range(Ua.shape[1]):
        u = Ua[:, r]
        v = Va[:, r]
        w = Wa[:, r]
        ui = np.nonzero(u)[0]
        vi = np.nonzero(v)[0]
        wi = np.nonzero(w)[0]
        for a in ui:
            for b in vi:
                for c in wi:
                    T[a, b, c] += u[a] * v[b] * w[c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # Try multiple block splits and return the best valid decomposition
    candidates = []

    # Split 1: 4x4x4 + 4x4x4 + 4x4x1 (rank 114)
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    col = _add_444(U, V, W, col, 0, 0, 0)
    col = _add_444(U, V, W, col, 0, 4, 4)
    col = _schoolbook_block(U, V, W, col, 0, 8, 8, 4, 4, 1)
    if col == rank and _verify(U, V, W, n, m, p):
        candidates.append((rank, {"U": U, "V": V, "W": W}))

    # Split 2: 4x4x3 + 4x4x3 + 4x4x3 (schoolbook each: rank 48 each, total 144)
    # But try Strassen-like for 4x4x3
    # For now, just use schoolbook for comparison
    rank = 144
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    col = _schoolbook_block(U, V, W, col, 0, 0, 0, 4, 4, 3)
    col = _schoolbook_block(U, V, W, col, 0, 3, 3, 4, 4, 3)
    col = _schoolbook_block(U, V, W, col, 0, 6, 6, 4, 4, 3)
    if col == rank and _verify(U, V, W, n, m, p):
        candidates.append((rank, {"U": U, "V": V, "W": W}))

    # Split 3: 4x4x5 + 4x4x4 (try to find good 4x4x5)
    # 4x4x5: split into 4x4x2 + 4x4x3
    # 4x4x2 schoolbook: 32, 4x4x3 schoolbook: 48, total 80 for 4x4x5
    # Plus 4x4x4 Strassen: 49, total 129
    # Not better than 114

    # Return best candidate or fallback
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    # Fallback: schoolbook rank 144
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
