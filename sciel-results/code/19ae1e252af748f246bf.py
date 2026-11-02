import numpy as np
import time
import itertools
from itertools import combinations

DIMENSIONS = [2, 4, 5]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # ---- Build the rank-36 block-Strassen base as a list of term dicts ----
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_block(y_indices, z_indices, w_indices):
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        w00, w01, w10, w11 = w_indices

        # M1 = (y00+y11)(z00+z11)
        add_term({u_idx(*y00): 1, u_idx(*y11): 1},
                 {v_idx(*z00): 1, v_idx(*z11): 1},
                 {w_idx(*w00): 1, w_idx(*w11): 1})
        # M2 = (y10+y11)z00
        add_term({u_idx(*y10): 1, u_idx(*y11): 1},
                 {v_idx(*z00): 1},
                 {w_idx(*w10): 1, w_idx(*w11): -1})
        # M3 = y00(z01-z11)
        add_term({u_idx(*y00): 1},
                 {v_idx(*z01): 1, v_idx(*z11): -1},
                 {w_idx(*w01): 1, w_idx(*w11): 1})
        # M4 = y11(z10-z00)
        add_term({u_idx(*y11): 1},
                 {v_idx(*z10): 1, v_idx(*z00): -1},
                 {w_idx(*w00): 1, w_idx(*w10): 1})
        # M5 = (y00+y01)z11
        add_term({u_idx(*y00): 1, u_idx(*y01): 1},
                 {v_idx(*z11): 1},
                 {w_idx(*w00): -1, w_idx(*w01): 1})
        # M6 = (y10-y00)(z00+z01)
        add_term({u_idx(*y10): 1, u_idx(*y00): -1},
                 {v_idx(*z00): 1, v_idx(*z10 if False else z01): 1},
                 {w_idx(*w11): 1})
        # M7 = (y01-y11)(z10+z11)
        add_term({u_idx(*y01): 1, u_idx(*y11): -1},
                 {v_idx(*z10): 1, v_idx(*z11): 1},
                 {w_idx(*w00): 1})

    def schoolbook_2x2x1(y_indices, z_indices, w_indices):
        y00, y01, y10, y11 = y_indices
        z0, z1 = z_indices
        w0, w1 = w_indices

        add_term({u_idx(*y00): 1}, {v_idx(*z0): 1}, {w_idx(*w0): 1})
        add_term({u_idx(*y01): 1}, {v_idx(*z1): 1}, {w_idx(*w0): 1})
        add_term({u_idx(*y10): 1}, {v_idx(*z0): 1}, {w_idx(*w1): 1})
        add_term({u_idx(*y11): 1}, {v_idx(*z1): 1}, {w_idx(*w1): 1})

    yL = [(0, 0), (0, 1), (1, 0), (1, 1)]
    yR = [(0, 2), (0, 3), (1, 2), (1, 3)]

    zA = [(0, 0), (0, 1), (1, 0), (1, 1)]
    zB = [(2, 0), (2, 1), (3, 0), (3, 1)]
    zC = [(0, 2), (0, 3), (1, 2), (1, 3)]
    zD = [(2, 2), (2, 3), (3, 2), (3, 3)]

    wA = [(0, 0), (1, 0), (0, 1), (1, 1)]
    wB = [(2, 0), (3, 0), (2, 1), (3, 1)]

    zA1 = [(0, 4), (1, 4)]
    zB1 = [(2, 4), (3, 4)]
    wC = [(4, 0), (4, 1)]

    strassen_block(yL, zA, wA)
    strassen_block(yR, zB, wA)
    strassen_block(yL, zC, wB)
    strassen_block(yR, zD, wB)
    schoolbook_2x2x1(yL, zA1, wC)
    schoolbook_2x2x1(yR, zB1, wC)

    R = len(terms)

    # ---- Verification machinery (exact tensor check) ----
    def verify(kept):
        # sum over kept terms of (u_r outer v_r) @ w_r == matmul tensor
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for r in kept:
            u, v, w = terms[r]
            for (iu, cu) in u.items():
                for (iv, cv) in v.items():
                    for (iw, cw) in w.items():
                        T[iu, iv, iw] += cu * cv * cw
        # target tensor: T[i*m+j, j*p+k, k*n+i] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                        return False
        # all other entries must be 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i, j = divmod(a, m)
                    j2, k = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if not (i == i2 and j == j2 and k == k2):
                        if abs(T[a, b, c]) > 1e-9:
                            return False
        return True

    # ---- Try to drop a single redundant term ----
    best = list(range(R))
    if time.time() < deadline - 30:
        for r in range(R):
            cand = [x for x in range(R) if x != r]
            if verify(cand):
                best = cand
                break

    # ---- Try to drop a pair of terms (cheap: only if single-drop failed) ----
    if time.time() < deadline - 20:
        for r, s in combinations(range(R), 2):
            cand = [x for x in range(R) if x != r and x != s]
            if verify(cand):
                best = cand
                break

    # ---- Build final matrices from best kept set ----
    rank = len(best)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    for new_r, old_r in enumerate(best):
        u, v, w = terms[old_r]
        for (iu, cu) in u.items():
            U[iu, new_r] = cu
        for (iv, cv) in v.items():
            V[iv, new_r] = cv
        for (iw, cw) in w.items():
            W[iw, new_r] = cw

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
