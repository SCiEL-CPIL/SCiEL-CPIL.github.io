"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7-mult coefficients for X*Y=Z
    # P1 = (x11+x22)(y11+y22)
    # P2 = (x21+x22)y11
    # P3 = x11(y12-y22)
    # P4 = x22(y21-y11)
    # P5 = (x11+x12)y22
    # P6 = (x21-x11)(y11+y12)
    # P7 = (x12-x22)(y21+y22)
    
    # Reconstruction:
    # z11 = P1 + P4 - P5 + P7
    # z12 = P3 + P5
    # z21 = P2 + P4
    # z22 = P1 - P2 + P3 + P6
    
    # Define the 7 products as lists of (x_idx, coeff) and (y_idx, coeff)
    # x_idx: 0:x11, 1:x12, 2:x21, 3:x22
    # y_idx: 0:y11, 1:y12, 2:y21, 3:y22
    
    products = [
        # P1
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        # P2
        ([ (2, 1), (3, 1) ], [ (0, 1) ]),
        # P3
        ([ (0, 1) ], [ (1, 1), (3, -1) ]),
        # P4
        ([ (3, 1) ], [ (2, 1), (0, -1) ]),
        # P5
        ([ (0, 1), (1, 1) ], [ (3, 1) ]),
        # P6
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ]),
        # P7
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ])
    ]
    
    # Reconstruction coefficients for z11, z12, z21, z22 from P1..P7
    # z11: P1(1), P4(1), P5(-1), P7(1)
    # z12: P3(1), P5(1)
    # z21: P2(1), P4(1)
    # z22: P1(1), P2(-1), P3(1), P6(1)
    recon = [
        [1, 0, 0, 1, -1, 0, 1], # z11
        [0, 0, 1, 0, 1, 0, 0],  # z12
        [0, 1, 0, 1, 0, 0, 0],  # z21
        [1, -1, 1, 0, 0, 1, 0]  # z22
    ]
    
    def add_strassen_block(U_indices, V_indices, W_indices, col_offset):
        for p_idx in range(7):
            x_coeffs, y_coeffs = products[p_idx]
            for x_i, x_c in x_coeffs:
                U[U_indices[x_i]][col_offset + p_idx] += x_c
            for y_i, y_c in y_coeffs:
                V[V_indices[y_i]][col_offset + p_idx] += y_c
            
            # Update W
            for z_i in range(4):
                coeff = recon[z_i][p_idx]
                if coeff != 0:
                    W[W_indices[z_i]][col_offset + p_idx] += coeff

    # Block 1: A1 * B11 -> C11
    # A1 indices in U: x11=0, x12=1, x21=4, x22=5
    U_idx_1 = [0, 1, 4, 5]
    # B11 indices in V: y11=0, y12=1, y21=4, y22=5
    V_idx_1 = [0, 1, 4, 5]
    # C11 indices in W: z11=0, z12=2, z21=1, z22=3
    W_idx_1 = [0, 2, 1, 3]
    add_strassen_block(U_idx_1, V_idx_1, W_idx_1, 0)

    # Block 2: A2 * B21 -> C11
    # A2 indices in U: x11=2, x12=3, x21=6, x22=7
    U_idx_2 = [2, 3, 6, 7]
    # B21 indices in V: y11=8, y12=9, y21=12, y22=13
    V_idx_2 = [8, 9, 12, 13]
    # C11 indices in W: same as above
    W_idx_2 = [0, 2, 1, 3]
    add_strassen_block(U_idx_2, V_idx_2, W_idx_2, 7)

    # Block 3: A1 * B12 -> C12
    # A1 indices in U: same as U_idx_1
    U_idx_3 = [0, 1, 4, 5]
    # B12 indices in V: y11=2, y12=3, y21=6, y22=7
    V_idx_3 = [2, 3, 6, 7]
    # C12 indices in W: z11=4, z12=6, z21=5, z22=7
    W_idx_3 = [4, 6, 5, 7]
    add_strassen_block(U_idx_3, V_idx_3, W_idx_3, 14)

    # Block 4: A2 * B22 -> C12
    # A2 indices in U: same as U_idx_2
    U_idx_4 = [2, 3, 6, 7]
    # B22 indices in V: y11=10, y12=11, y21=14, y22=15
    V_idx_4 = [10, 11, 14, 15]
    # C12 indices in W: same as W_idx_3
    W_idx_4 = [4, 6, 5, 7]
    add_strassen_block(U_idx_4, V_idx_4, W_idx_4, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
