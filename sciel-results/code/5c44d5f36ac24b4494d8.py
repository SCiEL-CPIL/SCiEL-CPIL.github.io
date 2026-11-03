import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen 2x2x2: M1..M7
    # U (X part) per s: list of (i_loc, j_loc, coeff)
    U_s = [
        [(0,0,1),(1,1,1)],   # M1
        [(1,0,1),(1,1,1)],   # M2
        [(0,0,1)],           # M3
        [(1,1,1)],           # M4
        [(0,0,1),(0,1,1)],   # M5
        [(1,0,1),(0,0,-1)],  # M6
        [(0,1,1),(1,1,-1)],  # M7
    ]
    # V (Y part) per s: (j_loc, k_loc, coeff)
    V_s = [
        [(0,0,1),(1,1,1)],   # M1
        [(0,0,1)],           # M2
        [(0,1,1),(1,1,-1)],  # M3
        [(1,0,1),(0,0,-1)],  # M4
        [(1,1,1)],           # M5
        [(0,0,1),(0,1,1)],   # M6
        [(1,0,1),(1,1,1)],   # M7
    ]
    # W (Z part) per s: (i_loc, k_loc, coeff)
    W_s = [
        [(0,0,1),(1,1,1)],   # M1
        [(1,0,1),(1,1,-1)],  # M2
        [(0,1,1),(1,1,1)],   # M3
        [(0,0,1),(1,0,1)],   # M4
        [(0,0,-1),(0,1,1)],  # M5
        [(1,1,1)],           # M6
        [(0,0,1)],           # M7
    ]

    col = 0
    for b in range(3):
        for a in range(2):
            for s in range(7):
                for (i_loc, j_loc, c) in U_s[s]:
                    U[i_loc * m + a * 2 + j_loc, col] += c
                for (j_loc, k_loc, c) in V_s[s]:
                    V[(a * 2 + j_loc) * p + b * 2 + k_loc, col] += c
                for (i_loc, k_loc, c) in W_s[s]:
                    W[(b * 2 + k_loc) * n + i_loc, col] += c
                col += 1

    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W],
    }
