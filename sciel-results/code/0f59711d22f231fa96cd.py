# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We decompose the multiplication A(2x4) * B(4x8) -> C(2x8)
    # using block-level Strassen.
    # Split j (inner dim) into 2 groups of size 2.
    # Split k (outer dim) into 4 groups of size 2.
    # For each j-group and k-group, we perform a 2x2 * 2x2 multiplication
    # using Strassen (rank 7).
    # Total blocks: 2 (j) * 4 (k) = 8 blocks.
    # Total rank: 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # (8, 56)
    V = [[0] * rank for _ in range(m * p)]  # (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # (16, 56)
    
    # Strassen coefficients for 2x2 * 2x2
    # Indices: 0,1 for rows/cols of sub-blocks
    # P: A sub-block (2x2), Q: B sub-block (2x2), R: C sub-block (2x2)
    
    # P coefficients: term_idx -> { (i, j): coeff }
    P_coeff = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): 1},       # M2
        {(0,0): 1},                  # M3
        {(1,1): 1},                  # M4
        {(0,0): 1, (0,1): 1},       # M5
        {(1,0): 1, (0,0): -1},      # M6
        {(0,1): 1, (1,1): -1},      # M7
    ]
    
    # Q coefficients: term_idx -> { (j, k): coeff }
    Q_coeff = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                  # M2
        {(0,1): 1, (1,1): -1},      # M3
        {(1,0): 1, (0,0): -1},      # M4
        {(1,1): 1},                  # M5
        {(0,0): 1, (0,1): 1},       # M6
        {(1,0): 1, (1,1): 1},       # M7
    ]
    
    # R coefficients: term_idx -> { (i, k): coeff }
    R_coeff = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): -1},      # M2
        {(0,1): 1, (1,1): 1},       # M3
        {(0,0): 1, (1,0): 1},       # M4
        {(0,0): -1, (0,1): 1},      # M5
        {(1,1): 1},                  # M6
        {(0,0): 1},                  # M7
    ]
    
    col = 0
    # Iterate over j-groups (2 groups of 2)
    for jg in range(2):
        j_offset = jg * 2
        # Iterate over k-groups (4 groups of 2)
        for kg in range(4):
            k_offset = kg * 2
            # Iterate over Strassen terms
            for t in range(7):
                # Update U
                for (i_loc, j_loc), c_p in P_coeff[t].items():
                    i_global = i_loc
                    j_global = j_offset + j_loc
                    U[i_global * m + j_global][col] += c_p
                
                # Update V
                for (j_loc, k_loc), c_q in Q_coeff[t].items():
                    j_global = j_offset + j_loc
                    k_global = k_offset + k_loc
                    V[j_global * p + k_global][col] += c_q
                
                # Update W
                for (i_loc, k_loc), c_r in R_coeff[t].items():
                    i_global = i_loc
                    k_global = k_offset + k_loc
                    W[k_global * n + i_global][col] += c_r
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
