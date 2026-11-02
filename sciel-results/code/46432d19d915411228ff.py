import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Inner Strassen 2x2 definitions
    # u: list of (row, col, sign)
    # v: list of (row, col, sign)
    # w: list of (row, col, sign)
    inner_defs = [
        {'u': [(0,0,1),(1,1,1)], 'v': [(0,0,1),(1,1,1)], 'w': [(0,0,1),(1,1,1)]},
        {'u': [(1,0,1),(1,1,1)], 'v': [(0,0,1)], 'w': [(1,0,1),(1,1,-1)]},
        {'u': [(0,0,1)], 'v': [(0,1,1),(1,1,-1)], 'w': [(0,1,1),(1,1,1)]},
        {'u': [(1,1,1)], 'v': [(1,0,1),(0,0,-1)], 'w': [(0,0,1),(1,0,1)]},
        {'u': [(0,0,1),(0,1,1)], 'v': [(1,1,1)], 'w': [(0,1,1),(0,0,-1)]},
        {'u': [(1,0,1),(0,0,-1)], 'v': [(0,0,1),(0,1,1)], 'w': [(1,1,1)]},
        {'u': [(0,1,1),(1,1,-1)], 'v': [(1,0,1),(1,1,1)], 'w': [(0,0,1)]},
    ]

    # Outer Strassen 4x4 definitions (2x2 blocks of 2x2)
    outer_defs = [
        {'a': [(0,0,1),(1,1,1)], 'b': [(0,0,1),(1,1,1)], 'c': [(0,0,1),(1,1,1)]},
        {'a': [(1,0,1),(1,1,1)], 'b': [(0,0,1)], 'c': [(1,0,1),(1,1,-1)]},
        {'a': [(0,0,1)], 'b': [(0,1,1),(1,1,-1)], 'c': [(0,1,1),(1,1,1)]},
        {'a': [(1,1,1)], 'b': [(1,0,1),(0,0,-1)], 'c': [(0,0,1),(1,0,1)]},
        {'a': [(0,0,1),(0,1,1)], 'b': [(1,1,1)], 'c': [(0,1,1),(0,0,-1)]},
        {'a': [(1,0,1),(0,0,-1)], 'b': [(0,0,1),(0,1,1)], 'c': [(1,1,1)]},
        {'a': [(0,1,1),(1,1,-1)], 'b': [(1,0,1),(1,1,1)], 'c': [(0,0,1)]},
    ]

    for s, outer in enumerate(outer_defs):
        for t, inner in enumerate(inner_defs):
            r = s * 7 + t
            
            # U contributions
            for (P, Q, sign_a) in outer['a']:
                for (ir, jc, sign_x) in inner['u']:
                    coeff = sign_a * sign_x
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        u_idx = row * m + col
                        U[u_idx, r] += coeff

            # V contributions
            for (P, Q, sign_b) in outer['b']:
                for (ir, jc, sign_y) in inner['v']:
                    coeff = sign_b * sign_y
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        v_idx = row * p + col
                        V[v_idx, r] += coeff

            # W contributions
            for (P, Q, sign_c) in outer['c']:
                for (ir, jc, sign_z) in inner['w']:
                    coeff = sign_c * sign_z
                    if coeff != 0:
                        row = 2 * P + ir
                        col = 2 * Q + jc
                        w_idx = col * n + row
                        W[w_idx, r] += coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
