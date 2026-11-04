import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    # Split A (2x3) into AL (2x2) and AR (2x1)
    # Split B (3x6) into BT (2x6) and BB (1x6)
    # C = AL * BT + AR * BB
    
    # AL * BT: 2x2 * 2x6
    # Split BT into 3 blocks of 2x2: BT0, BT1, BT2
    # Use Strassen for each 2x2*2x2
    
    # AR * BB: 2x1 * 1x6
    # Outer product, rank 12
    
    total_rank = 33
    U = [[0] * total_rank for _ in range(n * m)] # 6 x 33
    V = [[0] * total_rank for _ in range(m * p)] # 18 x 33
    W = [[0] * total_rank for _ in range(p * n)] # 12 x 33
    
    col = 0
    
    # Strassen terms for 2x2 * 2x2
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y00-y01)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # Z00 = M1 + M4 - M5 - M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # A indices for AL (cols 0,1 of A):
    # A00: idx 0, A01: idx 1, A10: idx 3, A11: idx 4
    # B indices for BT (rows 0,1 of B):
    # For block t, cols 2t, 2t+1
    # B00_t: idx 0*6 + 2t, B01_t: idx 0*6 + 2t+1
    # B10_t: idx 1*6 + 2t, B11_t: idx 1*6 + 2t+1
    # C indices for block t (cols 2t, 2t+1 of C):
    # C00_t: idx (2t)*2 + 0, C01_t: idx (2t+1)*2 + 0
    # C10_t: idx (2t)*2 + 1, C11_t: idx (2t+1)*2 + 1
    
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for i, val in u_vals.items():
            U[i][col] += val
        for i, val in v_vals.items():
            V[i][col] += val
        for i, val in w_vals.items():
            W[i][col] += val
        col += 1

    for t in range(3):
        # B indices
        b00 = 0 * 6 + 2 * t
        b01 = 0 * 6 + 2 * t + 1
        b10 = 1 * 6 + 2 * t
        b11 = 1 * 6 + 2 * t + 1
        
        # C indices
        c00 = (2 * t) * 2 + 0
        c01 = (2 * t + 1) * 2 + 0
        c10 = (2 * t) * 2 + 1
        c11 = (2 * t + 1) * 2 + 1
        
        # A indices (fixed for AL)
        a00 = 0
        a01 = 1
        a10 = 3
        a11 = 4
        
        # M1
        add_term(
            {a00: 1, a11: 1},
            {b00: 1, b11: 1},
            {c00: 1, c11: 1}
        )
        # M2
        add_term(
            {a10: 1, a11: 1},
            {b00: 1},
            {c10: 1}
        )
        # M3
        add_term(
            {a00: 1},
            {b10: 1, b11: -1},
            {c01: 1, c11: 1}
        )
        # M4
        add_term(
            {a11: 1},
            {b00: 1, b01: -1},
            {c00: 1, c10: 1}
        )
        # M5
        add_term(
            {a00: 1, a01: 1},
            {b11: 1},
            {c00: -1, c01: 1}
        )
        # M6
        add_term(
            {a10: 1, a00: -1},
            {b00: 1, b01: 1},
            {c11: 1}
        )
        # M7
        add_term(
            {a01: 1, a11: -1},
            {b10: 1, b11: 1},
            {c00: -1}
        )
        
    # AR * BB part
    # AR is A[:, 2] -> indices 2, 5
    # BB is B[2, :] -> indices 12, 13, 14, 15, 16, 17
    # C is 2x6
    # C0j = A02 * B2j
    # C1j = A12 * B2j
    
    for j in range(6):
        # Term for C0j
        a_idx = 2  # A02
        b_idx = 12 + j # B2j
        c_idx = j * 2 + 0 # C0j
        add_term(
            {a_idx: 1},
            {b_idx: 1},
            {c_idx: 1}
        )
        
        # Term for C1j
        a_idx = 5  # A12
        b_idx = 12 + j # B2j
        c_idx = j * 2 + 1 # C1j
        add_term(
            {a_idx: 1},
            {b_idx: 1},
            {c_idx: 1}
        )
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
