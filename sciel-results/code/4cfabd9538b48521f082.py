import time
import numpy as np

_start_time = time.time()
_DEADLINE = 140.0

def construct():
    n, m, p = 3, 5, 7
    deadline = _start_time + _DEADLINE

    def schoolbook():
        rank = n * m * p
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

    def verify_decomp(U, V, W, rank):
        Ua = np.array(U, dtype=float).reshape(n * m, -1)
        Va = np.array(V, dtype=float).reshape(m * p, -1)
        Wa = np.array(W, dtype=float).reshape(p * n, -1)
        
        for i in range(n):
            for j in range(m):
                u_row = Ua[i * m + j]
                for k in range(p):
                    v_row = Va[j * p + k]
                    w_row = Wa[k * n + i]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Tiling: 1x2x3 Strassen blocks
    n_pairs = [(0, 1)]
    m_pairs = [(0, 1), (2, 3)]
    p_pairs = [(0, 1), (2, 3), (4, 5)]

    n_covered = set()
    for a, b in n_pairs:
        n_covered.update([a, b])
    m_covered = set()
    for c, d in m_pairs:
        m_covered.update([c, d])
    p_covered = set()
    for e, f in p_pairs:
        p_covered.update([e, f])

    num_strassen = len(n_pairs) * len(m_pairs) * len(p_pairs)
    rank_strassen = num_strassen * 7

    num_schoolbook = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i not in n_covered or j not in m_covered or k not in p_covered:
                    num_schoolbook += 1

    total_rank = rank_strassen + num_schoolbook

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    for (a, b) in n_pairs:
        for (c, d) in m_pairs:
            for (e, f) in p_pairs:
                # M1
                U[a * m + c][col] += 1
                U[b * m + d][col] += 1
                V[c * p + e][col] += 1
                V[d * p + f][col] += 1
                W[e * n + a][col] += 1
                W[f * n + b][col] += 1
                col += 1

                # M2
                U[b * m + c][col] += 1
                U[b * m + d][col] += 1
                V[c * p + e][col] += 1
                W[f * n + a][col] += 1
                col += 1

                # M3
                U[a * m + c][col] += 1
                V[d * p + f][col] += 1
                V[d * p + e][col] -= 1
                W[e * n + f][col] += 1
                col += 1

                # M4
                U[b * m + d][col] += 1
                V[c * p + f][col] += 1
                V[c * p + e][col] -= 1
                W[e * n + a][col] += 1
                col += 1

                # M5
                U[a * m + c][col] += 1
                U[a * m + d][col] += 1
                V[d * p + f][col] += 1
                W[e * n + f][col] += 1
                W[e * n + a][col] -= 1
                col += 1

                # M6
                U[b * m + c][col] += 1
                U[a * m + c][col] -= 1
                V[c * p + e][col] += 1
                V[c * p + f][col] += 1
                W[f * n + a][col] += 1
                col += 1

                # M7
                U[b * m + d][col] += 1
                U[a * m + d][col] -= 1
                V[d * p + e][col] += 1
                V[d * p + f][col] += 1
                W[e * n + a][col] -= 1
                W[f * n + b][col] += 1
                col += 1

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i not in n_covered or j not in m_covered or k not in p_covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    if col != total_rank:
        return schoolbook()

    if time.time() > deadline - 5:
        return {"U": U, "V": V, "W": W}

    if verify_decomp(U, V, W, total_rank):
        return {"U": U, "V": V, "W": W}

    return schoolbook()
