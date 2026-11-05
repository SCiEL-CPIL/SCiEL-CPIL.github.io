import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook fallback
    def _schoolbook():
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

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Build 28-term Strassen basis for 2x2x8 (rows 1,2 of A and B)
    rank_s = 28
    U_s = [[0] * rank_s for _ in range(n * m)]
    V_s = [[0] * rank_s for _ in range(m * p)]
    W_s = [[0] * rank_s for _ in range(p * n)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    # la=0 corresponds to global a=1, la=1 to global a=2
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_s[li * m + (1 + la)][col] = u_val
            for la in range(2):
                for lk in range(2):
                    # la=0 corresponds to global a=1, la=1 to global a=2
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_s[(1 + la) * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_s[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Build 29-term decomposition (Strassen 28 + 1 term for a=0)
    rank_29 = 29
    U29 = [[0] * rank_29 for _ in range(n * m)]
    V29 = [[0] * rank_29 for _ in range(m * p)]
    W29 = [[0] * rank_29 for _ in range(p * n)]

    # Copy Strassen part
    for r in range(rank_s):
        for i in range(n):
            for a in range(1, 3): # a=1,2
                U29[i * m + a][r] = U_s[i * m + a][r]
        for a in range(1, 3): # a=1,2
            for k in range(p):
                V29[a * p + k][r] = V_s[a * p + k][r]
        for k in range(p):
            for i in range(n):
                W29[k * n + i][r] = W_s[k * n + i][r]

    # Add 1 term for a=0
    r_new = rank_s
    for i in range(n):
        U29[i * m + 0][r_new] = 1
    for k in range(p):
        V29[0 * p + k][r_new] = 1
    for k in range(p):
        for i in range(n):
            W29[k * n + i][r_new] = 1

    candidate = {"U": U29, "V": V29, "W": W29}
    if _validate(candidate):
        return candidate
    
    # Fallback if validation fails (should not happen)
    return _schoolbook()

# EVOLVE-BLOCK-END
