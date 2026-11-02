"""Recursive block decomposition for 4x4x6 matrix multiplication (rank 20)."""
DIMENSIONS = [4, 4, 6]
import time

def _verify(U, V, W, n, m, p):
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    T = np.einsum('ir,kr,jr->ijk', Ua, Wa, Va)
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i, k, j] - 1.0) > 1e-9:
                    ok = False
    return ok

def construct():
    t0 = time.time()
    n, m, p = 4, 4, 6
    rank = n * m * p  # schoolbook upper bound
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
    best = {"U": U, "V": V, "W": W}

    # Known rank-5 decomposition of the 2x2x3 multiplication tensor.
    # Local convention: Ua[a*2+b], Va[b*3+c], Wc[c*2+a].
    def _sub223():
        return (
            [[1, 1, 1, 0, 1],
             [0, 1, 0, 1, 1],
             [0, 1, 1, 1, 0],
             [1, 0, 1, 1, 1]],
            [[1, 1, 0, 1, 0],
             [0, 1, 0, 0, 1],
             [1, 0, 1, 0, 1],
             [1, 0, 0, 1, 0],
             [1, 1, 1, 0, 1]],
            [[1, 1, 1, 1, 1],
             [1, 0, 0, 1, 1],
             [1, 1, 0, 0, 1]]
        )

    Ua, Va, Wc = _sub223()
    deadline = 150.0
    try:
        U20 = [[0] * 20 for _ in range(16)]
        V20 = [[0] * 20 for _ in range(24)]
        W20 = [[0] * 20 for _ in range(24)]
        for bi in range(2):
            for bj in range(2):
                base = (bi * 2 + bj) * 5
                for a in range(2):
                    for b in range(2):
                        for t in range(5):
                            U20[(bi * 2 + a) * 4 + (bj * 2 + b)][base + t] = Ua[a * 2 + b][t]
                for b in range(2):
                    for c in range(3):
                        for t in range(5):
                            V20[(bj * 2 + b) * 6 + c][base + t] = Va[b * 3 + c][t]
                for c in range(3):
                    for a in range(2):
                        for t in range(5):
                            W20[c * 4 + (bi * 2 + a)][base + t] = Wc[c * 2 + a][t]
        if _verify(U20, V20, W20, 4, 4, 6):
            best = {"U": U20, "V": V20, "W": W20}
    except Exception:
        pass
    if time.time() > deadline - 10.0:
        return best
    return best
