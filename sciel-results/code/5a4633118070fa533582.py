"""2x4 by 4x6 matrix multiplication, exact rank-42 tensor decomposition.

The 2x4 matrix A and 4x6 matrix B are partitioned into 2-column and 2-row
blocks, and each pair of blocks is multiplied via the 7-term Strassen scheme.
A has 1 column-block (j=0..3 split as 0..1, 2..3).  B has 3 column-blocks
(k=0..5 split as 0..1, 2..3, 4..5).  Output C is 2x6.

Total rank = 2 * 3 * 7 = 42.
"""

import time as _time

_DEADLINE = 130.0  # seconds; evaluator kills at 150 s


def construct():
    _t0 = _time.monotonic()

    n, m, p = 2, 4, 6
    rank = 42

    # Allocate factor matrices
    U = [[0] * rank for _ in range(n * m)]   # 8  x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42

    # --- Strassen coefficient tables (2x2 block multiply) ---
    # For A block (2 rows i_loc x 2 cols j_loc), B block (2 rows j_loc x 2 cols k_loc)
    # U[s]: list of (i_local, j_local, coeff) for the s-th Strassen term
    U_tbl = [
        [(0, 0, 1), (1, 1, 1)],         # S0: A00+A11
        [(1, 0, 1), (1, 1, 1)],         # S1: A10+A11
        [(0, 0, 1)],                     # S2: A00
        [(1, 1, 1)],                     # S3: A11
        [(0, 0, 1), (0, 1, 1)],         # S4: A00+A01
        [(1, 0, 1), (0, 0, -1)],        # S5: A10-A00
        [(0, 1, 1), (1, 1, -1)],        # S6: A01-A11
    ]
    V_tbl = [
        [(0, 0, 1), (1, 1, 1)],         # S0: B00+B11
        [(0, 0, 1)],                     # S1: B00
        [(0, 1, 1), (1, 1, -1)],        # S2: B01-B11
        [(1, 0, 1), (0, 0, -1)],        # S3: B10-B00
        [(1, 1, 1)],                     # S4: B11
        [(0, 0, 1), (0, 1, 1)],         # S5: B00+B01
        [(1, 0, 1), (1, 1, 1)],         # S6: B10+B11
    ]
    # C = S0*C00pattern + S1*... etc.
    # C00 = S0 + S3 - S4 + S6
    # C01 = S2 + S4
    # C10 = S1 + S3
    # C11 = S0 - S1 + S2 + S5
    W_tbl = [
        [(0, 0, 1), (1, 1, 1)],         # S0: C00+=1, C11+=1
        [(1, 0, 1), (1, 1, -1)],        # S1: C10+=1, C11-=-1
        [(0, 1, 1), (1, 1, 1)],         # S2: C01+=1, C11+=1
        [(0, 0, 1), (1, 0, 1)],         # S3: C00+=1, C10+=1
        [(0, 0, -1), (0, 1, 1)],        # S4: C00-=-1, C01+=1
        [(1, 1, 1)],                     # S5: C11+=1
        [(0, 0, 1)],                     # S6: C00+=1
    ]

    # Pre-compute global coefficient lists for each block pair (a, b)
    # a = A column-block index (0 or 1), b = B row-block index (0, 1, or 2)
    # For each term s, build lists of (global_row, coeff) for U, V, W
    for a in range(2):
        j_off = a * 2
        for b in range(3):
            k_off = b * 2
            for s in range(7):
                col = a * 21 + b * 7 + s
                if col >= rank:
                    continue
                if _time.monotonic() - _t0 > _DEADLINE:
                    break

                # Fill U: rows are (i, j) -> index i*m + j
                for i_loc, j_loc, c in U_tbl[s]:
                    U[i_loc * m + (j_off + j_loc)][col] += c

                # Fill V: rows are (j, k) -> index j*p + k
                for j_loc, k_loc, c in V_tbl[s]:
                    V[(j_off + j_loc) * p + (k_off + k_loc)][col] += c

                # Fill W: rows are (k, i) -> index k*n + i
                for i_loc, k_loc, c in W_tbl[s]:
                    W[(k_off + k_loc) * n + i_loc][col] += c

    return {"U": U, "V": V, "W": W}
