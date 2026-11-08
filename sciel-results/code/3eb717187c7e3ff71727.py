"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42: 2 A-column-blocks (2x2) x 3 B-column-blocks (2x2) x 7 Strassen terms
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen coefficients for 2x2 matrix M (U factor)
    # M indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # s=0: M00 + M11
        {(1, 0): 1, (1, 1): 1},   # s=1: M10 + M11
        {(0, 0): 1},               # s=2: M00
        {(1, 1): 1},               # s=3: M11
        {(0, 0): 1, (0, 1): 1},   # s=4: M00 + M01
        {(1, 0): 1, (0, 0): -1},  # s=5: M10 - M00
        {(0, 1): 1, (1, 1): -1}   # s=6: M01 - M11
    ]
    
    # Strassen coefficients for 2x2 matrix N (V factor)
    # N indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},   # s=0: N00 + N11
        {(0, 0): 1},               # s=1: N00
        {(0, 1): 1, (1, 1): -1},  # s=2: N01 - N11
        {(1, 0): 1, (0, 0): -1},  # s=3: N10 - N00
        {(1, 1): 1},               # s=4: N11
        {(0, 0): 1, (0, 1): 1},   # s=5: N00 + N01
        {(1, 0): 1, (1, 1): 1}    # s=6: N10 + N11
    ]
    
    # Strassen coefficients for 2x2 output R (W factor)
    # R indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # R00 = P0 + P3 - P4 + P6
    # R01 = P2 + P4
    # R10 = P1 + P3
    # R11 = P0 - P1 + P2 + P5
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},    # s=0 (P0): R00+1, R11+1
        {(1, 0): 1, (1, 1): -1},   # s=1 (P1): R10+1, R11-1
        {(0, 1): 1, (1, 1): 1},    # s=2 (P2): R01+1, R11+1
        {(0, 0): 1, (1, 0): 1},    # s=3 (P3): R00+1, R10+1
        {(0, 0): -1, (0, 1): 1},   # s=4 (P4): R00-1, R01+1
        {(1, 1): 1},                # s=5 (P5): R11+1
        {(0, 0): 1}                 # s=6 (P6): R00+1
    ]
    
    col = 0
    for a in range(2):  # A column-block: 0 for cols {0,1}, 1 for cols {2,3}
        for b in range(3):  # B column-block: 0 for cols {0,1}, 1 for {2,3}, 2 for {4,5}
            for s in range(7):  # Strassen term
                # Update U: A[i, a*2 + j_loc]
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V: B[a*2 + j_loc, b*2 + k_loc]
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W: C[b*2 + k_loc, i_loc] -> W[k*n + i]
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
