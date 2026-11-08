"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses a block decomposition with Strassen:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114
"""

import time

DIMENSIONS = [4, 4, 9]

# Strassen 2x2*2x2: 7 terms, each (A_coeffs, B_coeffs, C_coeffs)
# A: a=(0,0), b=(0,1), c=(1,0), d=(1,1)
# B: e=(0,0), f=(0,1), g=(1,0), h=(1,1)
# C: c11=(0,0), c12=(0,1), c21=(1,0), c22=(1,1)
STRASSEN_2X2 = [
    # s1 = (a+d)(e+h) -> c11:+1, c22:+1
    ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    # s2 = (c+d)e -> c21:+1, c22:-1
    ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
    # s3 = a(f-h) -> c12:+1, c22:+1
    ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
    # s4 = d(g-e) -> c11:+1, c21:+1
    ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
    # s5 = (a+b)h -> c11:-1, c12:+1
    ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
    # s6 = (c-a)(e+f) -> c22:+1
    ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
    # s7 = (b-d)(g+h) -> c11:+1
    ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
]

# Block-level Strassen for 4x4*4x4 (2x2 blocks of 2x2)
# M1 = (A11+A22)(B11+B22) -> C11:+1, C22:+1
# M2 = (A21+A22)B11       -> C21:+1, C22:-1
# M3 = A11(B12-B22)       -> C12:+1, C22:+1
# M4 = A22(B21-B11)       -> C11:+1, C21:+1
# M5 = (A11+A12)B22       -> C11:-1, C12:+1
# M6 = (A21-A11)(B11+B12) -> C22:+1
# M7 = (A12-A22)(B21+B22) -> C11:+1
BLOCK_STRASSEN = [
    # (A_side, B_side, C_side)
    # A_side: list of (block_row, block_col, coeff)
    # B_side: list of (block_row, block_col, coeff)
    # C_side: dict (block_row, block_col) -> coeff
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], {(0,0):1, (1,1):1}),   # M1
    ([(1,0,1),(1,1,1)], [(0,0,1)],       {(1,0):1, (1,1):-1}),    # M2
    ([(0,0,1)],       [(0,1,1),(1,1,-1)],{(0,1):1, (1,1):1}),     # M3
    ([(1,1,1)],       [(1,0,1),(0,0,-1)],{(0,0):1, (1,0):1}),     # M4
    ([(0,0,1),(0,1,1)],[(1,1,1)],        {(0,0):-1,(0,1):1}),     # M5
    ([(1,0,1),(0,0,-1)],[(0,0,1),(0,1,1)],{(1,1):1}),             # M6
    ([(0,1,1),(1,1,-1)],[(1,0,1),(1,1,1)],{(0,0):1}),             # M7
]


def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    num_A = n * m   # 16
    num_B = m * p   # 36
    num_C = p * n   # 36
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """Add 49 Strassen terms for 4x4 * 4x4."""
        for A_side, B_side, C_side in BLOCK_STRASSEN:
            for A_meta, B_meta, C_meta in STRASSEN_2X2:
                u_vec = [0] * num_A
                v_vec = [0] * num_B
                w_vec = [0] * num_C
                
                for (li, lj), mc in A_meta.items():
                    for (br, bc, bc_coeff) in A_side:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += mc * bc_coeff
                
                for (lj, lk), mc in B_meta.items():
                    for (br, bc, bc_coeff) in B_side:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += mc * bc_coeff
                
                for (li, lk), mc in C_meta.items():
                    for (br, bc), bc_coeff in C_side.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += mc * bc_coeff
                
                add_term(u_vec, v_vec, w_vec)
    
    # Part 1: A * B[0:4] (4x4 * 4x4) -> C[0:4]
    strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B[4:8] (4x4 * 4x4) -> C[4:8]
    strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B[8:9] (4x4 * 4x1) -> C[8:9], schoolbook 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * num_A
            v_vec = [0] * num_B
            w_vec = [0] * num_C
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
    
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(num_A)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(num_B)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(num_C)]
    
    return {"U": U, "V": V, "W": W}
