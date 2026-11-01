"""
Matrix multiplication 2x4 * 4x4 tensor decomposition.
Uses Strassen's algorithm on 2x2 blocks to achieve rank 28.
"""

import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin within 120s

    n, m, p = 2, 4, 4
    rank = 28

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    def idx_a(i, j):
        return i * m + j

    def idx_b(i, j):
        return i * p + j

    def idx_c(i, j):
        # W[k*n+i, r] where i is row of C, k is col of C
        return j * n + i

    def add_strassen_block(offset, a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        u_00 = idx_a(a_r0, a_c0)
        u_01 = idx_a(a_r0, a_c0 + 1)
        u_10 = idx_a(a_r0 + 1, a_c0)
        u_11 = idx_a(a_r0 + 1, a_c0 + 1)

        v_00 = idx_b(b_r0, b_c0)
        v_01 = idx_b(b_r0, b_c0 + 1)
        v_10 = idx_b(b_r0 + 1, b_c0)
        v_11 = idx_b(b_r0 + 1, b_c0 + 1)

        w_00 = idx_c(c_r0, c_c0)
        w_01 = idx_c(c_r0, c_c0 + 1)
        w_10 = idx_c(c_r0 + 1, c_c0)
        w_11 = idx_c(c_r0 + 1, c_c0 + 1)

        # M1 = (a00+a11)(b00+b11)
        U[u_00, offset + 0] += 1
        U[u_11, offset + 0] += 1
        V[v_00, offset + 0] += 1
        V[v_11, offset + 0] += 1
        W[w_00, offset + 0] += 1
        W[w_11, offset + 0] += 1

        # M2 = (a10+a11)b00
        U[u_10, offset + 1] += 1
        U[u_11, offset + 1] += 1
        V[v_00, offset + 1] += 1
        W[w_10, offset + 1] += 1
        W[w_11, offset + 1] -= 1

        # M3 = a00(b01-b11)
        U[u_00, offset + 2] += 1
        V[v_01, offset + 2] += 1
        V[v_11, offset + 2] -= 1
        W[w_01, offset + 2] += 1
        W[w_11, offset + 2] += 1

        # M4 = a11(b10-b00)
        U[u_11, offset + 3] += 1
        V[v_10, offset + 3] += 1
        V[v_00, offset + 3] -= 1
        W[w_00, offset + 3] += 1
        W[w_10, offset + 3] += 1

        # M5 = (a00+a01)b11
        U[u_00, offset + 4] += 1
        U[u_01, offset + 4] += 1
        V[v_11, offset + 4] += 1
        W[w_00, offset + 4] -= 1
        W[w_01, offset + 4] += 1

        # M6 = (a10-a00)(b00+b01)
        U[u_10, offset + 5] += 1
        U[u_00, offset + 5] -= 1
        V[v_00, offset + 5] += 1
        V[v_01, offset + 5] += 1
        W[w_11, offset + 5] += 1

        # M7 = (a01-a11)(b10+b11)
        U[u_01, offset + 6] += 1
        U[u_11, offset + 6] -= 1
        V[v_10, offset + 6] += 1
        V[v_11, offset + 6] += 1
        W[w_00, offset + 6] += 1

    # Four 2x2 block multiplications
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
