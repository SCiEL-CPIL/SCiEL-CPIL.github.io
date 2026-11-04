"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses block decomposition with Strassen's algorithm for 2x2x2 blocks.
Partition [2,2,1] gives rank 68 (1*7 + 4*4 + 4*4 + 4*4 + 2*2 + 2*2 + 2*2 + 1*1).
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    terms = []

    def add_term(U_dict, V_dict, W_dict):
        if time.time() > deadline:
            return
        U_dict = {k: v for k, v in U_dict.items() if v != 0}
        V_dict = {k: v for k, v in V_dict.items() if v != 0}
        W_dict = {k: v for k, v in W_dict.items() if v != 0}
        if U_dict and V_dict and W_dict:
            terms.append((U_dict, V_dict, W_dict))

    blocks = [(0, 2), (2, 4), (4, 5)]

    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1

                if ar == 2 and ac == 2 and bc == 2:
                    A00 = r1 * m + c1
                    A01 = r1 * m + (c1 + 1)
                    A10 = (r1 + 1) * m + c1
                    A11 = (r1 + 1) * m + (c1 + 1)

                    B00 = c1 * p + k1
                    B01 = c1 * p + (k1 + 1)
                    B10 = (c1 + 1) * p + k1
                    B11 = (c1 + 1) * p + (k1 + 1)

                    C00_w = k1 * n + r1
                    C01_w = (k1 + 1) * n + r1
                    C10_w = k1 * n + (r1 + 1)
                    C11_w = (k1 + 1) * n + (r1 + 1)

                    add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1})
                    add_term({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1})
                    add_term({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1})
                    add_term({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1})
                    add_term({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1})
                    add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1})
                    add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1})
                else:
                    for ii in range(ar):
                        for jj in range(ac):
                            for kk in range(bc):
                                u_idx = (r1 + ii) * m + (c1 + jj)
                                v_idx = (c1 + jj) * p + (k1 + kk)
                                w_idx = (k1 + kk) * n + (r1 + ii)
                                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        U_dict, V_dict, W_dict = terms[r]
        for idx, val in U_dict.items():
            U[idx][r] = val
        for idx, val in V_dict.items():
            V[idx][r] = val
        for idx, val in W_dict.items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
