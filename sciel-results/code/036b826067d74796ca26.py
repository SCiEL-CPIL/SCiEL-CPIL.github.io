import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 140.0

    n, m, p = 2, 2, 6

    # Rank 21 decomposition using 3 blocks of Strassen (7 terms each)
    rank = 21
    U_arr = np.zeros((n * m, rank), dtype=int)
    V_arr = np.zeros((m * p, rank), dtype=int)
    W_arr = np.zeros((p * n, rank), dtype=int)

    # Strassen terms for A (2x2)
    # Indices: 0:A00, 1:A01, 2:A10, 3:A11
    U_terms = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]

    # Strassen terms for B (2x2 block)
    # For a block with global columns [c0, c1]:
    # Local col 0 -> global c0, Local col 1 -> global c1
    # V indices: j*p + k
    V_terms_block = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B0,c0 + B1,c1
        [(0, 0, 1)],                 # M2: B0,c0
        [(0, 1, 1), (1, 1, -1)],     # M3: B0,c1 - B1,c1
        [(1, 0, 1), (0, 0, -1)],     # M4: B1,c0 - B0,c0
        [(1, 1, 1)],                 # M5: B1,c1
        [(0, 0, 1), (0, 1, 1)],      # M6: B0,c0 + B0,c1
        [(1, 0, 1), (1, 1, 1)]       # M7: B1,c0 + B1,c1
    ]

    # Strassen terms for C (2x2 block)
    # For a block with output cols [d0, d1]:
    # Local col 0 -> global d0, Local col 1 -> global d1
    # W indices: k*n + i
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_terms_block = [
        [(0, 0, 1), (1, 1, 1)],   # M1: C0,d0(1), C1,d1(1)
        [(1, 0, 1), (1, 1, -1)],  # M2: C1,d0(1), C1,d1(-1)
        [(0, 1, 1), (1, 1, 1)],   # M3: C0,d1(1), C1,d1(1)
        [(0, 0, 1), (1, 0, 1)],   # M4: C0,d0(1), C1,d0(1)
        [(0, 0, -1), (0, 1, 1)],  # M5: C0,d0(-1), C0,d1(1)
        [(1, 1, 1)],              # M6: C1,d1(1)
        [(0, 0, 1)]               # M7: C0,d0(1)
    ]

    for block in range(3):
        col_off = 2 * block
        for t in range(7):
            r = block * 7 + t
            for idx, val in U_terms[t].items():
                U_arr[idx, r] = val
            for j_loc, k_loc, val in V_terms_block[t]:
                gk = col_off + k_loc
                V_arr[j_loc * p + gk, r] = val
            for i, k_loc, val in W_terms_block[t]:
                gk = col_off + k_loc
                W_arr[gk * n + i, r] = val

    U = U_arr.tolist()
    V = V_arr.tolist()
    W = W_arr.tolist()

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
