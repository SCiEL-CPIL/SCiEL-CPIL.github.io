"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank 88 decomposition:
    # 2x M_2,4,4 (rank 28 each via Strassen) + 2x M_1,4,4 (rank 16 each via schoolbook)
    n, m, p = DIMENSIONS
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2 multiplication
    # Each factor is (a_coeffs, b_coeffs, c_coeffs) where coeffs map (row, col) -> value
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_factors = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process 2x M_2,4,4 blocks (for rows i=0,1 of the 3x4 matrix)
    # Each M_2,4,4 uses Strassen on 2x2 sub-blocks: 4 sub-products of 2x2x2
    for c_block in range(2):
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            
            # 4 sub-products: (A_left * B_LL), (A_left * B_LR), (A_right * B_RL), (A_right * B_RR)
            for sub_prod_idx in range(4):
                if sub_prod_idx == 0:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 0
                    c_i_off, c_j_off = 0, 0
                elif sub_prod_idx == 1:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 2
                    c_i_off, c_j_off = 0, 2
                elif sub_prod_idx == 2:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 0
                    c_i_off, c_j_off = 0, 0
                else:  # sub_prod_idx == 3
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2
                
                for (ai, aj), au in a_coeffs.items():
                    u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                    U[u_idx][col] += au
                
                for (bi, bj), bv in b_coeffs.items():
                    v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                    V[v_idx][col] += bv
                
                for (ci, cj), cw in c_coeffs.items():
                    w_idx = (cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)
                    W[w_idx][col] += cw
                
                col += 1
    
    # Process 2x M_1,4,4 blocks (for row i=2 of the 3x4 matrix)
    # Schoolbook: 16 terms per block
    for c_block in range(2):
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k
                v_idx = k * 8 + (j + 4 * c_block)
                w_idx = (j + 4 * c_block) * 3 + 2
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
