"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen parameters for 2x2x2
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22) B11
    # P3 = A11 (B12 - B22)
    # P4 = A22 (B21 - B11)
    # P5 = (A11 + A12) B22
    # P6 = (A21 - A11) (B11 + B12)
    # P7 = (A12 - A22) (B21 + B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # A blocks: (i, j_offset)
    # A is 2x4.
    # For a 2x2 sub-problem, A is 2x2.
    # We have 2 such A blocks (left and right).
    # A_left: cols 0,1. A_right: cols 2,3.
    
    # B blocks: (j_offset, k_offset)
    # B is 4x8.
    # We have 2 row blocks (top, bottom) and 4 col blocks (c0-c1, c2-c3, c4-c5, c6-c7).
    
    # W blocks: (i, k_offset)
    # W is 8x2 (transposed C).
    # We have 4 col blocks.
    
    # Strassen Coeffs
    # A coeffs: dict (i, j) -> val
    # B coeffs: dict (j, k) -> val
    # W coeffs: dict (k, i) -> val
    
    strassen = [
        # P1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # P3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # P4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # P5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # P6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col = 0
    # Iterate over 4 output column pairs (k_block) and 2 intermediate row pairs (j_block)
    for k_block in range(4):
        for j_block in range(2):
            # A block is determined by j_block
            # j_block 0 -> A cols 0,1
            # j_block 1 -> A cols 2,3
            a_col_offset = j_block * 2
            
            # B block is determined by j_block and k_block
            # j_block 0 -> B rows 0,1
            # j_block 1 -> B rows 2,3
            b_row_offset = j_block * 2
            
            # k_block 0 -> B cols 0,1
            # k_block 1 -> B cols 2,3
            # k_block 2 -> B cols 4,5
            # k_block 3 -> B cols 6,7
            b_col_offset = k_block * 2
            
            # W block is determined by k_block
            # k_block 0 -> W cols 0,1 (which are C rows 0,1? No, W is p x n = 8 x 2)
            # W[k*n + i].
            # k_block 0 -> k=0,1. W indices 0*2+0, 0*2+1, 1*2+0, 1*2+1.
            w_col_offset = k_block * 2
            
            for u_c, v_c, w_c in strassen:
                # Fill U
                for (i, j), coeff in u_c.items():
                    u_idx = i * m + (a_col_offset + j)
                    U[u_idx][col] += coeff
                
                # Fill V
                for (j, k), coeff in v_c.items():
                    v_idx = (b_row_offset + j) * p + (b_col_offset + k)
                    V[v_idx][col] += coeff
                
                # Fill W
                for (k, i), coeff in w_c.items():
                    w_idx = (w_col_offset + k) * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
