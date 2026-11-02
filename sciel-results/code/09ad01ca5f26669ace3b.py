"""Matrix multiplication 2x4 x 4x4 tensor decomposition - rank 28 Strassen construction."""

import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for b in range(2):
        for a_blk in range(2):
            for s in range(7):
                c = col
                j0 = 2 * a_blk
                j1 = 2 * a_blk + 1

                # U contributions: U[i * 4 + j, c] += coeff
                U_contrib = {
                    0: [(0, j0, 1), (1, j1, 1)],
                    1: [(1, j0, 1), (1, j1, 1)],
                    2: [(0, j0, 1)],
                    3: [(1, j1, 1)],
                    4: [(0, j0, 1), (0, j1, 1)],
                    5: [(1, j0, 1), (0, j0, -1)],
                    6: [(0, j1, 1), (1, j1, -1)],
                }
                for (i, j, coeff) in U_contrib[s]:
                    U[i * 4 + j, c] += coeff

                # V contributions: V[j * 4 + k, c] += coeff
                V_contrib = {
                    0: [(2 * a_blk, 2 * b, 1), (2 * a_blk + 1, 2 * b + 1, 1)],
                    1: [(2 * a_blk, 2 * b, 1)],
                    2: [(2 * a_blk, 2 * b + 1, 1), (2 * a_blk + 1, 2 * b + 1, -1)],
                    3: [(2 * a_blk + 1, 2 * b, 1), (2 * a_blk, 2 * b, -1)],
                    4: [(2 * a_blk + 1, 2 * b + 1, 1)],
                    5: [(2 * a_blk, 2 * b, 1), (2 * a_blk, 2 * b + 1, 1)],
                    6: [(2 * a_blk + 1, 2 * b, 1), (2 * a_blk + 1, 2 * b + 1, 1)],
                }
                for (j, k, coeff) in V_contrib[s]:
                    V[j * 4 + k, c] += coeff

                # W contributions: W[(2*b + k_loc) * 2 + i_loc, c] += coeff
                # W index: k * n + i where k = 2*b + k_loc, i = i_loc
                W_contrib = {
                    0: {(0, 0): 1, (1, 1): 1},
                    1: {(1, 0): 1, (1, 1): -1},
                    2: {(0, 1): 1, (1, 1): 1},
                    3: {(0, 0): 1, (1, 0): 1},
                    4: {(0, 0): -1, (0, 1): 1},
                    5: {(1, 1): 1},
                    6: {(0, 0): 1},
                }
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    W[(2 * b + k_loc) * 2 + i_loc, c] += coeff

                col += 1

    # Return as lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
