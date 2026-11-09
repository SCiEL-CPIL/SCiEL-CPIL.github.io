import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 97

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen 2x2x2 inner coefficients
    u_in = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_in = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_out = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    u_out = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_out = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_outer = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    col = 0
    for k in range(7):
        for l in range(7):
            u_vec = np.zeros(16, dtype=int)
            v_vec = np.zeros(28, dtype=int)
            w_vec = np.zeros(28, dtype=int)

            for i_inner in range(2):
                for j_inner in range(2):
                    ci = u_in[l][i_inner * 2 + j_inner]
                    if ci == 0:
                        continue
                    for b in range(4):
                        co = u_out[k][b]
                        if co == 0:
                            continue
                        b_row = b // 2
                        b_col = b % 2
                        i_global = 2 * b_row + i_inner
                        j_global = 2 * b_col + j_inner
                        u_vec[i_global * 4 + j_global] += ci * co

            for i_inner in range(2):
                for j_inner in range(2):
                    ci = v_in[l][i_inner * 2 + j_inner]
                    if ci == 0:
                        continue
                    for b in range(4):
                        co = v_out[k][b]
                        if co == 0:
                            continue
                        b_row = b // 2
                        b_col = b % 2
                        j_global = 2 * b_row + i_inner
                        k_global = 2 * b_col + j_inner
                        v_vec[j_global * 7 + k_global] += ci * co

            for c_block in range(4):
                co_w = w_outer[c_block][k]
                if co_w == 0:
                    continue
                for c_inner in range(4):
                    ci_w = w_out[c_inner][l]
                    if ci_w == 0:
                        continue
                    tc = co_w * ci_w
                    if tc == 0:
                        continue
                    cb_row = c_block // 2
                    cb_col = c_block % 2
                    ci_row = c_inner // 2
                    ci_col = c_inner % 2
                    i_global = 2 * cb_row + ci_row
                    k_global = 2 * cb_col + ci_col
                    w_vec[k_global * 4 + i_global] += tc

            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1

    # Schoolbook for the 4x4 * 4x3 part
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                k_global = 4 + k_prime
                U[i * 4 + j, col] = 1
                V[j * 7 + k_global, col] = 1
                W[k_global * 4 + i, col] = 1
                col += 1

    # Convert to lists of lists with integer values
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
