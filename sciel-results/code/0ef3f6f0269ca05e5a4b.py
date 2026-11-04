"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # Rank-56 decomposition using 2 blocks x 4 subblocks x 7 Strassen terms
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen coefficients for 7 terms
    # Each entry: (u_a00, u_a01, u_a10, u_a11, v_b00, v_b01, v_b10, v_b11, w_c00, w_c01, w_c10, w_c11)
    strassen = [
        ( 1,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1),  # M1
        ( 0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  1,  0),  # M2
        ( 1,  0,  0,  0,  0,  1,  0, -1,  0,  1,  0,  0),  # M3
        ( 0,  0,  0,  1,  0,  0,  1, -1, -1,  0,  1,  0),  # M4
        ( 1,  1,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0),  # M5
        ( 0,  0,  1, -1,  1,  1,  0,  0,  0,  0,  0,  1),  # M6
        ( 0,  1,  0, -1,  0,  0,  1,  1, -1,  0,  0,  1),  # M7
    ]
    
    for b in range(2):      # block: 0 = A_L/B_T, 1 = A_R/B_B
        for m_idx in range(4):  # subblock: which 2x2 block of B/C
            for s in range(7):   # Strassen term
                r = b * 28 + m_idx * 7 + s
                
                coeff = strassen[s]
                u_a00, u_a01, u_a10, u_a11 = coeff[0], coeff[1], coeff[2], coeff[3]
                v_b00, v_b01, v_b10, v_b11 = coeff[4], coeff[5], coeff[6], coeff[7]
                w_c00, w_c01, w_c10, w_c11 = coeff[8], coeff[9], coeff[10], coeff[11]
                
                # A indices: row i ∈ {0,1}, col = 2*b + {0,1}
                # U[i*m + j, r] where i=row, j=col of A
                a_col_base = 2 * b
                U[0 * m + (a_col_base + 0)][r] = u_a00  # A[0, 2b]
                U[0 * m + (a_col_base + 1)][r] = u_a01  # A[0, 2b+1]
                U[1 * m + (a_col_base + 0)][r] = u_a10  # A[1, 2b]
                U[1 * m + (a_col_base + 1)][r] = u_a11  # A[1, 2b+1]
                
                # B indices: row j = 2*b + {0,1}, col = 2*m_idx + {0,1}
                # V[j*p + k, r] where j=row, k=col of B
                b_row_base = 2 * b
                b_col_base = 2 * m_idx
                V[(b_row_base + 0) * p + (b_col_base + 0)][r] = v_b00  # B[2b, 2m]
                V[(b_row_base + 0) * p + (b_col_base + 1)][r] = v_b01  # B[2b, 2m+1]
                V[(b_row_base + 1) * p + (b_col_base + 0)][r] = v_b10  # B[2b+1, 2m]
                V[(b_row_base + 1) * p + (b_col_base + 1)][r] = v_b11  # B[2b+1, 2m+1]
                
                # W indices: W[k*n + i, r] where i=row of C, k=col of C
                # C[0, 2m], C[0, 2m+1], C[1, 2m], C[1, 2m+1]
                c_col_base = 2 * m_idx
                W[(c_col_base + 0) * n + 0][r] = w_c00  # C[0, 2m]
                W[(c_col_base + 1) * n + 0][r] = w_c01  # C[0, 2m+1]
                W[(c_col_base + 0) * n + 1][r] = w_c10  # C[1, 2m]
                W[(c_col_base + 1) * n + 1][r] = w_c11  # C[1, 2m+1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
