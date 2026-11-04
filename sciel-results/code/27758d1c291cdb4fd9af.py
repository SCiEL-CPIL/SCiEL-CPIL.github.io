import time
from fractions import Fraction
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = 0
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        uv = u_row[c] * v_row[c]
                        if uv != 0:
                            acc += uv * w_row[c]
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        acc += S_U[r][i*2 + a] * S_V[r][a*2 + k] * S_W[r][k*2 + i]
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        # Fallback to schoolbook if Strassen fails (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Rank 36: 4 blocks of (Strassen 2x2x2 + Schoolbook 2x1x2)
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        
        # Part 1: Strassen for A[:2, :2] * B[:2, k0:k0+2]
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li*2 + la]
                    if u_val:
                        U[li * m + la][col] += u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la*2 + lk]
                    if v_val:
                        V[la * p + (k0 + lk)][col] += v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk*2 + li]
                    if w_val:
                        W[(k0 + lk) * n + li][col] += w_val
            col += 1
            if not _elapsed_ok():
                # Return current best if timeout? Or fallback?
                # Fallback to schoolbook is safer if we haven't finished
                pass 

        # Part 2: Schoolbook for A[:2, 2] * B[2, k0:k0+2]
        for li in range(2):
            for lk in range(2):
                U[li * m + 2][col] = 1
                V[2 * p + (k0 + lk)][col] = 1
                W[(k0 + lk) * n + li][col] = 1
                col += 1
                if not _elapsed_ok():
                    pass

    candidate = {"U": U, "V": V, "W": W}
    if _validate(candidate):
        return candidate
    
    # Fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for a in range(m):
            for k in range(p):
                U[i * m + a][col] = 1
                V[a * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
