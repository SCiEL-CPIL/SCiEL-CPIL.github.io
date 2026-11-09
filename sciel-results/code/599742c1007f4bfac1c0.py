import time
import random

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s margin for 120s limit
    
    n, m, p = 2, 4, 5
    
    # We construct the decomposition using 2x2x2 Strassen blocks and 2x2x1 schoolbook blocks.
    # The problem 2x4x5 is split into two 2x2x5 sub-problems (rows of B split into two groups of 2).
    # Each 2x2x5 is split into one 2x2x2 and one 2x2x3.
    # 2x2x2 uses Strassen (rank 7).
    # 2x2x3 uses 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11.
    # Total rank per 2x2x5 block: 7 + 11 = 18.
    # Total rank: 18 + 18 = 36.
    
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    # Strassen coefficients for 2x2x2
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # r=0..6
    S_U = [
        [1,0,0,1],  # M1
        [0,0,1,1],  # M2
        [1,0,0,0],  # M3
        [0,0,0,1],  # M4
        [1,1,0,0],  # M5
        [-1,0,1,0], # M6
        [0,1,0,-1]  # M7
    ]
    S_V = [
        [1,0,0,1],  # M1
        [1,0,0,0],  # M2
        [0,1,0,-1], # M3
        [-1,0,1,0], # M4
        [0,0,0,1],  # M5
        [1,1,0,0],  # M6
        [0,0,1,1]   # M7
    ]
    S_W = [
        [1,0,0,1],  # M1
        [0,1,0,-1], # M2
        [0,0,1,1],  # M3
        [1,1,0,0],  # M4
        [-1,0,1,0], # M5
        [0,0,0,1],  # M6
        [1,0,0,0]   # M7
    ]
    
    # Schoolbook coefficients for 2x2x1
    # U[i*2+j], V[j*2+0], W[0*2+i]
    # r=0..3
    # M1: A00 B00
    # M2: A01 B10
    # M3: A10 B00
    # M4: A11 B10
    SB_U = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]
    SB_V = [
        [1,0],
        [0,1],
        [1,0],
        [0,1]
    ]
    SB_W = [
        [1,0],
        [1,0],
        [0,1],
        [0,1]
    ]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen_block(i_base, j_base, k_base, col_start):
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi * m + gj][col_start + r] = S_U[r][li * 2 + lj]
            for lj in range(2):
                for lk in range(2):
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj * p + gk][col_start + r] = S_V[r][lj * 2 + lk]
            for lk in range(2):
                for li in range(2):
                    gk = k_base + lk
                    gi = i_base + li
                    W[gk * n + gi][col_start + r] = S_W[r][lk * 2 + li]
        return col_start + 7

    # Helper to add Schoolbook 2x2x1 block
    def add_schoolbook_2x2x1_block(i_base, j_base, k_base, col_start):
        for r in range(4):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi * m + gj][col_start + r] = SB_U[r][li * 2 + lj]
            for lj in range(2):
                gj = j_base + lj
                gk = k_base
                V[gj * p + gk][col_start + r] = SB_V[r][lj]
            for li in range(2):
                gk = k_base
                gi = i_base + li
                W[gk * n + gi][col_start + r] = SB_W[r][li]
        return col_start + 4

    # We have two main blocks corresponding to j_base = 0 and j_base = 2.
    # For each j_block, we have 5 k columns.
    # We split k into k0=0,1 (Strassen) and k1=2,3,4 (Schoolbook 2x2x3).
    # Schoolbook 2x2x3 is split into 2x2x2 (Strassen) + 2x2x1 (Schoolbook).
    # So for each j_block:
    # 1. Strassen for k=0,1 (j_block, k_base=0)
    # 2. Strassen for k=2,3 (j_block, k_base=2)
    # 3. Schoolbook for k=4 (j_block, k_base=4)
    
    # Block 1: j_base = 0
    col = add_strassen_block(0, 0, 0, col) # k=0,1
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_strassen_block(0, 0, 2, col) # k=2,3
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook_2x2x1_block(0, 0, 4, col) # k=4
    
    # Block 2: j_base = 2
    col = add_strassen_block(0, 2, 0, col) # k=0,1
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_strassen_block(0, 2, 2, col) # k=2,3
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook_2x2x1_block(0, 2, 4, col) # k=4
    
    assert col == rank, f"Column mismatch: {col} vs {rank}"
    
    return {"U": U, "V": V, "W": W}
