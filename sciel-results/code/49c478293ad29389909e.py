"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    # Rank-49 decomposition: pair the 14 columns of B into 7 (2x2) blocks.
    # Each block uses the rank-7 Strassen decomposition of 2x2 matrix multiplication.
    # Total rank = 7 blocks * 7 Strassen terms = 49 < 56 (schoolbook).
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    # Strassen coefficients for 2x2 multiplication.
    # Each term: (u_coeffs on A, v_coeffs on B, w_coeffs on C)
    # A indexed by (i,j) in {0,1}x{0,1} -> A flat index i*m+j = i*2+j
    # B indexed by (j,k) in {0,1}x{0,1} -> B flat index j*2+k (within block)
    # C indexed by (k,i) -> C flat index k*n+i = k*2+i
    #
    # Strassen 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11) B00
    # M3 = A00 (B10-B11)
    # M4 = A11 (B01-B10)
    # M5 = (A00+A01) B11
    # M6 = (A10-A00) (B00+B10)
    # M7 = (A01-A11) (B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # u: coefficients on A (4 entries: A00,A01,A10,A11)
    # v: coefficients on B (4 entries: B00,B01,B10,B11)
    # w: coefficients on C (4 entries: C00,C01,C10,C11)
    strassen = [
        # (u, v, w)
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 1]),
    ]
    # Let me re-derive carefully. C[k,i] where k is row, i is col.
    # C00 (k=0,i=0), C01 (k=0,i=1), C10 (k=1,i=0), C11 (k=1,i=1)
    # w index: k*n+i = k*2+i
    # C00 -> w[0], C01 -> w[1], C10 -> w[2], C11 -> w[3]
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So w for each term:
    # M1: C00+=1, C11+=1 -> [1,0,0,1]
    # M2: C10+=1, C11+=-1 -> [0,0,1,-1]
    # M3: C01+=1, C11+=1 -> [0,1,0,1]
    # M4: C00+=1, C10+=1 -> [1,0,1,0]
    # M5: C00+=-1, C01+=1 -> [-1,1,0,0]
    # M6: C11+=1 -> [0,0,0,1]
    # M7: C00+=1 -> [1,0,0,0]
    strassen = [
        # (u, v, w)
        # M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 1]),
        # M2 = (A10+A11) B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = A00 (B10-B11)
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        # M4 = A11 (B01-B10)
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),
        # M5 = (A00+A01) B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (A10-A00) (B00+B10)
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
        # M7 = (A01-A11) (B01+B11)
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),
    ]
    rank = num_blocks * len(strassen)
    U = [[0] * rank for _ in range(n * m)]  # 4 rows
    V = [[0] * rank for _ in range(m * p)]  # 28 rows
    W = [[0] * rank for _ in range(p * n)]  # 28 rows
    col = 0
    for b in range(num_blocks):
        for (u, v, w) in strassen:
            # A flat index: i*2+j
            for a_idx in range(4):
                U[a_idx][col] += u[a_idx]
            # B: block b, within-block index j*2+k -> global j*14 + (2b+k)
            for bv in range(4):
                j_in = bv // 2
                k_in = bv % 2
                global_b = j_in * p + (2 * b + k_in)
                V[global_b][col] += v[bv]
            # C: within-block index k*2+i -> global k*2 + i
            for cw in range(4):
                k_in = cw // 2
                i_in = cw % 2
                global_c = k_in * n + i_in
                W[global_c][col] += w[cw]
            col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
