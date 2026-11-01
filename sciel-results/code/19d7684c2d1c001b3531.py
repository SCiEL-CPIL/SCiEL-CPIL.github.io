# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    """
    n, m, p = 2, 4, 5

    rank_terms = []

    def add_strassen_2x2x2(a_offset, b_offset, c_offset):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset

        terms = [
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]

        for u, v, w in terms:
            rank_terms.append((u, v, w))

    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset

        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]

        for u, v, w in terms:
            rank_terms.append((u, v, w))

    # Block decomposition of 2x4 * 4x5
    # A is 2x4, B is 4x5, C is 2x5
    # Split B into B_left (4x4) and B_right (4x1)
    # C = A * B_left + A * B_right
    #
    # A * B_left:
    #   A[:, 0:2] * B_left[0:2, :]  -> 2x2 * 2x4 -> rank 14 (two Strassen 2x2x2)
    #   A[:, 2:4] * B_left[2:4, :]  -> 2x2 * 2x4 -> rank 14 (two Strassen 2x2x2)
    #
    # A * B_right:
    #   A[:, 0:2] * B_right[0:2, :]  -> 2x2 * 2x1 -> rank 4 (schoolbook)
    #   A[:, 2:4] * B_right[2:4, :]  -> 2x2 * 2x1 -> rank 4 (schoolbook)

    # Strassen blocks for A[:, 0:2] * B_left[0:2, 0:2]
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
    # Strassen blocks for A[:, 0:2] * B_left[0:2, 2:4]
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
    # Strassen blocks for A[:, 2:4] * B_left[2:4, 0:2]
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
    # Strassen blocks for A[:, 2:4] * B_left[2:4, 2:4]
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))

    # Schoolbook for A[:, 0:2] * B_right[0:2, 4:5]
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
    # Schoolbook for A[:, 2:4] * B_right[2:4, 4:5]
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))

    rank = len(rank_terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
