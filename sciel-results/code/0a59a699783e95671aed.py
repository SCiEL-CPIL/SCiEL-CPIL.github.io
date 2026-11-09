import time
import numpy as np

DIMENSIONS = [4, 4, 8]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Strassen rank-7 terms for 2x2x2 multiplication
    strassen_terms_2x2 = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1})
    ]

    top_level_terms = [
        ([((0, 0), 1), ((1, 1), 1)], [((0, 0), 1), ((1, 1), 1)], [((0, 0), 1), ((1, 1), 1)]),
        ([((1, 0), 1), ((1, 1), 1)], [((0, 0), 1)], [((1, 0), 1), ((1, 1), -1)]),
        ([((0, 0), 1)], [((0, 1), 1), ((1, 1), -1)], [((0, 1), 1), ((1, 1), 1)]),
        ([((1, 1), 1)], [((1, 0), 1), ((0, 0), -1)], [((0, 0), 1), ((1, 0), 1)]),
        ([((0, 0), 1), ((0, 1), 1)], [((1, 1), 1)], [((0, 0), -1), (0, 1), 1]),
        ([((1, 0), 1), ((0, 0), -1)], [((0, 0), 1), ((0, 1), 1)], [((1, 1), 1)]),
        ([((0, 1), 1), ((1, 1), -1)], [((1, 0), 1), ((1, 1), 1)], [((0, 0), 1)])
    ]
    # Fix term 4 W-part
    top_level_terms[4] = ([((0, 0), 1), ((0, 1), 1)], [((1, 1), 1)], [((0, 0), -1), ((0, 1), 1)])

    # Build 49 Strassen terms for 4x4x4
    # Each term: (u_dict, v_dict, w_dict) where keys are (row, col) pairs
    terms_4x4 = []
    for t_idx in range(7):
        a_blocks, b_blocks, c_blocks = top_level_terms[t_idx]
        for s_idx in range(7):
            s_u, s_v, s_w = strassen_terms_2x2[s_idx]

            u_global = {}
            for (r, c), coeff_s in s_u.items():
                for (u, v), coeff_a in a_blocks:
                    i = 2 * u + r
                    j = 2 * v + c
                    u_global[(i, j)] = u_global.get((i, j), 0) + coeff_s * coeff_a

            v_global = {}
            for (r, c), coeff_s in s_v.items():
                for (v, w), coeff_b in b_blocks:
                    j = 2 * v + r
                    k = 2 * w + c
                    v_global[(j, k)] = v_global.get((j, k), 0) + coeff_s * coeff_b

            w_global = {}
            for (r, c), coeff_s in s_w.items():
                for (u, w), coeff_c in c_blocks:
                    i = 2 * u + r
                    k = 2 * w + c
                    w_global[(i, k)] = w_global.get((i, k), 0) + coeff_s * coeff_c

            u_global = {k: v for k, v in u_global.items() if v != 0}
            v_global = {k: v for k, v in v_global.items() if v != 0}
            w_global = {k: v for k, v in w_global.items() if v != 0}

            if u_global and v_global and w_global:
                terms_4x4.append((u_global, v_global, w_global))

    rank = len(terms_4x4)

    # For 4x4 by 4x8: use dual numbers approach
    # B[j, k] = B_left[j, k%4] + eps * B_right[j, k%4]
    # U[i, j] is a scalar
    # Product: U[i,j] * B[j,k] = U[i,j]*B_left[j,k%4] + eps * U[i,j]*B_right[j,k%4]
    # W[i, k] = W_left[i, k%4] + eps * W_right[i, k%4]
    # Need: U[i,j]*B_left[j,k%4] = W_left[i, k%4]  (standard 4x4x4)
    #       U[i,j]*B_right[j,k%4] = W_right[i, k%4] (standard 4x4x4)
    # Both are satisfied by the same U if we use the same Strassen decomposition
    # V: V[j, k] = B[j, k%4] + eps*B[j, k%4] -> V[j, 0..3] = B_left, V[j, 4..7] = B_right
    # W: W[k, i] = W_left[i, k%4] + eps*W_right[i, k%4] -> W[k%4, i] and W[k%4+4, i]

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col_idx, (u_dict, v_dict, w_dict) in enumerate(terms_4x4):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col_idx] = coeff
        for (j, k), coeff in v_dict.items():
            # Left block: columns 0-3
            V[j * p + k][col_idx] = coeff
            # Right block: columns 4-7
            V[j * p + (k + 4)][col_idx] = coeff
        for (i, k), coeff in w_dict.items():
            # W index: k*n + i (trace convention W[k*n+i, r])
            W[k * n + i][col_idx] = coeff
            # Right half
            W[(k + 4) * n + i][col_idx] = coeff

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
