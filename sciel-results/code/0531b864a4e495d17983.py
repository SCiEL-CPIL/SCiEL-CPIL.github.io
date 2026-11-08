"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Strategy: Use Strassen (rank 49) for the two 4x4*4x4 blocks and schoolbook
(rank 16) for the final 4x4*4x1 block. Total rank = 49+49+16 = 114.

The key idea is a block decomposition:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 each use a rank-49 Strassen decomposition
  - A*B3 uses schoolbook with rank 16
  - Total rank: 114

Interface: construct() -> {'U', 'V', 'W'}
  U: (n*m, rank) = (16, rank)
  V: (m*p, rank) = (36, rank)
  W: (p*n, rank) = (36, rank)
  with W[k*n + i, r] following the official trace convention.
"""

import time

DIMENSIONS = [4, 4, 9]

# Strassen 2x2*2x2: 7 terms
# Each term: (A_coeffs_dict, B_coeffs_dict, C_coeffs_dict)
# A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
# B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
# C indices: (0,0)=c11, (0,1)=c12, (1,0)=c21, (1,1)=c22
STRASSEN_2X2 = [
    # s1 = (a+d)(e+h)
    ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
    # s2 = (c+d)e
    ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
    # s3 = a(f-h)
    ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
    # s4 = d(g-e)
    ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
    # s5 = (a+b)h
    ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
    # s6 = (c-a)(e+f)
    ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
    # s7 = (b-d)(g+h)
    ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
]

# Block-level Strassen for 4x4*4x4 (2x2 blocks of 2x2 matrices)
# Each entry: (A_block_terms, B_block_terms, C_block_dict)
# A_block_terms: list of (block_row, block_col, coeff)
# B_block_terms: list of (block_row, block_col, coeff)
# C_block_dict: (block_row, block_col) -> coeff
BLOCK_STRASSEN = [
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], {(0,0):1, (1,1):1}),
    ([(1,0,1),(1,1,1)], [(0,0,1)],       {(1,0):1, (1,1):-1}),
    ([(0,0,1)],       [(0,1,1),(1,1,-1)],{(0,1):1, (1,1):1}),
    ([(1,1,1)],       [(1,0,1),(0,0,-1)],{(0,0):1, (1,0):1}),
    ([(0,0,1),(0,1,1)],[(1,1,1)],        {(0,0):-1,(0,1):1}),
    ([(1,0,1),(0,0,-1)],[(0,0,1),(0,1,1)],{(1,1):1}),
    ([(0,1,1),(1,1,-1)],[(1,0,1),(1,1,1)],{(0,0):1}),
]


def _strassen_4x4_terms():
    """Generate the 49 U, V, W vectors for a single 4x4*4x4 Strassen block.
    
    Returns list of (u_vec, v_vec, w_vec) where:
      u_vec has length 16 (n*m for the A sub-block)
      v_vec has length 16 (m*p for the B sub-block)
      w_vec has length 16 (p*n for the C sub-block)
    """
    terms = []
    for A_side, B_side, C_side in BLOCK_STRASSEN:
        for A_meta, B_meta, C_meta in STRASSEN_2X2:
            u_vec = [0] * 16
            v_vec = [0] * 16
            w_vec = [0] * 16
            
            # Fill u_vec from A side
            for (li, lj), mc in A_meta.items():
                for (br, bc, bc_coeff) in A_side:
                    # Global index within the 4x4 A block: row = br*2+li, col = bc*2+lj
                    gi = br * 2 + li
                    gj = bc * 2 + lj
                    u_vec[gi * 4 + gj] += mc * bc_coeff
            
            # Fill v_vec from B side
            for (lj, lk), mc in B_meta.items():
                for (br, bc, bc_coeff) in B_side:
                    gj = br * 2 + lj
                    gk = bc * 2 + lk
                    v_vec[gj * 4 + gk] += mc * bc_coeff
            
            # Fill w_vec from C side (C index: [row=n, col=p] -> w[k*n+i] where i is row, k is col)
            for (li, lk), mc in C_meta.items():
                for (br, bc), bc_coeff in C_side.items():
                    gi = br * 2 + li  # this is the row index (i dimension)
                    gk = bc * 2 + lk  # this is the col index (k dimension)
                    w_vec[gk * 4 + gi] += mc * bc_coeff
            
            terms.append((u_vec, v_vec, w_vec))
    
    return terms


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

    def add_term(u_full, v_full, w_full):
        U_cols.append(u_full)
        V_cols.append(v_full)
        W_cols.append(w_full)

    # Precompute Strassen 4x4*4x4 terms (local 16x16x16)
    local_terms = _strassen_4x4_terms()

    # Part 1: A (4x4) * B[0:4] (4x4) -> C[0:4] (4x4)
    # A columns 0-3, B columns 0-3, C columns 0-3
    for u_loc, v_loc, w_loc in local_terms:
        u_full = [0] * num_A
        v_full = [0] * num_B
        w_full = [0] * num_C
        # Map local A index (row 0-3, col 0-3) to global
        for idx in range(16):
            u_full[idx] = u_loc[idx]
        # Map local B index (row 0-3, col 0-3) to global: col offset 0
        for idx in range(16):
            local_r = idx // 4
            local_c = idx % 4
            global_r = local_r
            global_c = local_c  # B columns 0-3
            v_full[global_r * p + global_c] = v_loc[idx]
        # Map local C index (k dimension 0-3, i dimension 0-3) to global
        for idx in range(16):
            local_k = idx // 4
            local_i = idx % 4
            global_k = local_k  # C columns 0-3
            global_i = local_i
            w_full[global_k * n + global_i] = w_loc[idx]
        add_term(u_full, v_full, w_full)

    if time.time() > deadline:
        # Return what we have so far
        pass

    # Part 2: A (4x4) * B[4:8] (4x4) -> C[4:8] (4x4)
    # A columns 0-3, B columns 4-7, C columns 4-7
    for u_loc, v_loc, w_loc in local_terms:
        u_full = [0] * num_A
        v_full = [0] * num_B
        w_full = [0] * num_C
        for idx in range(16):
            u_full[idx] = u_loc[idx]
        # Map local B index to global: col offset 4
        for idx in range(16):
            local_r = idx // 4
            local_c = idx % 4
            global_r = local_r
            global_c = local_c + 4
            v_full[global_r * p + global_c] = v_loc[idx]
        # Map local C index to global: col offset 4
        for idx in range(16):
            local_k = idx // 4
            local_i = idx % 4
            global_k = local_k + 4
            global_i = local_i
            w_full[global_k * n + global_i] = w_loc[idx]
        add_term(u_full, v_full, w_full)

    if time.time() > deadline:
        pass

    # Part 3: A (4x4) * B[8:9] (4x1) -> C[8:9] (4x1), schoolbook 16 terms
    for i in range(n):
        for j in range(m):
            u_full = [0] * num_A
            v_full = [0] * num_B
            w_full = [0] * num_C
            u_full[i * m + j] = 1
            v_full[j * p + 8] = 1
            w_full[8 * n + i] = 1
            add_term(u_full, v_full, w_full)

    if time.time() > deadline:
        pass

    rank = len(U_cols)

    # Fallback: schoolbook if something went wrong
    if rank == 0:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_full = [0] * num_A
                    v_full = [0] * num_B
                    w_full = [0] * num_C
                    u_full[i * m + j] = 1
                    v_full[j * p + k] = 1
                    w_full[k * n + i] = 1
                    add_term(u_full, v_full, w_full)
        rank = len(U_cols)

    # Build output matrices
    # U: (num_A, rank), V: (num_B, rank), W: (num_C, rank)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(num_A)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(num_B)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(num_C)]

    return {"U": U, "V": V, "W": W}
