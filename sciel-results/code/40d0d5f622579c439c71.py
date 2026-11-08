import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Returns a rank-189 exact tensor decomposition for 6x6 matrix multiplication
    using a 3x3 block partition where each 2x2 block multiplication uses Strassen's
    7-term algorithm: rank = 3^3 * 7 = 189.

    The factor matrices U (shape 36x189), V (shape 36x189), W (shape 36x189)
    satisfy: sum_k U[k] * V[k] * W[k] = matrix multiplication tensor.

    W follows the trace convention W[k*n+i, r] where k is the column of A's
    output row-block and i is the row of A's output.
    """
    deadline = time.time() + 120.0
    n, m, p = DIMENSIONS  # 6, 6, 6

    # Strassen's 7-term algorithm for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
    #
    # M1 = (a+d)(e+h)  -> C11 += M1, C22 += M1
    # M2 = (c+d)e      -> C21 += M2, C22 -= M2
    # M3 = a(f-h)      -> C12 += M3, C22 += M3
    # M4 = d(g-e)      -> C11 += M4, C21 += M4
    # M5 = (a+b)h      -> C11 -= M5, C12 += M5
    # M6 = (c-a)(e+f)  -> C22 += M6
    # M7 = (b-d)(g+h)  -> C11 += M7
    #
    # For each term, we store (U_coeffs, V_coeffs, W_coeffs) as dicts
    # U_coeffs: (row, col) -> value for the U factor (maps to A indices)
    # V_coeffs: (row, col) -> value for the V factor (maps to B indices)
    # W_coeffs: (row, col) -> value for the W factor (maps to C indices)
    #
    # In the 2x2 case, A is indexed (i, j) with i in {0,1} (row), j in {0,1} (col)
    # B is indexed (j, k) with j in {0,1} (row of B = col of A), k in {0,1} (col)
    # C is indexed (i, k) with i in {0,1} (row), k in {0,1} (col)
    #
    # For the 6x6 case, we tile into 3x3 blocks of 2x2 submatrices.
    # Block (I, J) of A occupies rows 2I..2I+1, cols 2J..2J+1
    # Block (J, K) of B occupies rows 2J..2J+1, cols 2K..2K+1
    # Block (I, K) of C occupies rows 2I..2I+1, cols 2K..2K+1

    strassen_patterns = [
        # M1 = (a+d)(e+h)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (c+d)*e
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = a*(f-h)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = d*(g-e)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (a+b)*h
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (c-a)*(e+f)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (b-d)*(g+h)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    rank = 27 * 7  # 189
    U = np.zeros((36, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))

    col_idx = 0
    for I in range(3):       # block row of A / block row of C
        for J in range(3):   # block col of A / block row of B
            for K in range(3):  # block col of B / block col of C
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    # U factor: maps to A's block (I, J)
                    # U row index: (2*I + pi) * 6 + (2*J + pj)
                    for (pi, pj), val in u_coeffs.items():
                        r = 2 * I + pi
                        c = 2 * J + pj
                        U[r * 6 + c, col_idx] += val
                    # V factor: maps to B's block (J, K)
                    # V row index: (2*J + pi) * 6 + (2*K + pj)
                    for (pi, pj), val in v_coeffs.items():
                        r = 2 * J + pi
                        c = 2 * K + pj
                        V[r * 6 + c, col_idx] += val
                    # W factor: maps to C's block (I, K)
                    # W uses trace convention: W[k*n + i, r]
                    # where i is the row index in C and k is the column index in C
                    # i = 2*I + pi, k = 2*K + pj
                    for (pi, pj), val in w_coeffs.items():
                        i = 2 * I + pi
                        k = 2 * K + pj
                        W[k * 6 + i, col_idx] += val
                    col_idx += 1

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return result
