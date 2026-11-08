import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 solution based on 2 blocks of Strassen 2x2x2
    # A is 2x4, B is 4x6.
    # Split A into A1 (2x2), A2 (2x2).
    # Split B into B1 (4x2), B2 (4x2), B3 (4x2).
    # C = A1 B1_top + A2 B1_bottom + ...
    # This is 6 separate 2x2x2 multiplications.
    # Rank = 6 * 7 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 coefficients
    # U: A (2x2)
    # V: B (2x2)
    # W: C (2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    U_s = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1}
    ]
    V_s = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1}
    ]
    W_s = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):-1},
        {(0,1):1, (1,1):1},
        {(0,0):1, (1,0):1},
        {(0,0):-1, (0,1):1},
        {(1,1):1},
        {(0,0):1}
    ]
    
    col = 0
    # A blocks: 0 (cols 0,1), 1 (cols 2,3)
    # B blocks: 0 (cols 0,1), 1 (cols 2,3), 2 (cols 4,5)
    # For each A block a and B block b, we do a 2x2x2 multiplication.
    # But wait, the B block is 2x2?
    # B is 4x6.
    # If we take A block 0 (2x2) and B block 0 (4x2), we need to split B block 0 into top 2x2 and bottom 2x2.
    # A = [A_top; A_bottom]? No, A is 2x4.
    # A = [A_L, A_R] where A_L is 2x2, A_R is 2x2.
    # B = [B_1; B_2] where B_1 is 2x6, B_2 is 2x6.
    # C = A_L B_1 + A_R B_2.
    # B_1 is 2x6. Split into B_11, B_12, B_13 (2x2 each).
    # B_2 is 2x6. Split into B_21, B_22, B_23 (2x2 each).
    # C_1 = A_L B_11 + A_R B_21.
    # C_2 = A_L B_12 + A_R B_22.
    # C_3 = A_L B_13 + A_R B_23.
    
    # So for each output column pair (k, k+1), we have:
    # Term 1: A_L * B_1k (2x2 * 2x2)
    # Term 2: A_R * B_2k (2x2 * 2x2)
    
    # Total 6 such multiplications.
    # Each has rank 7.
    # Total rank 42.
    
    for a_block in range(2): # 0 for A_L, 1 for A_R
        for b_block in range(3): # 0,1,2 for the 3 pairs of columns in B
            for s in range(7):
                # Update U
                # A index: i, j_local
                # j_global = a_block * 2 + j_local
                for (i_loc, j_loc), coeff in U_s[s].items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                # B index: j_local, k_local
                # j_global = a_block * 2 + j_local  (Wait, B rows correspond to A cols)
                # B row j_global corresponds to A col j_global.
                # So j_global is the same as above.
                # k_global = b_block * 2 + k_local
                for (j_loc, k_loc), coeff in V_s[s].items():
                    j = a_block * 2 + j_loc
                    k = b_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                # C index: i_loc, k_global
                # k_global = b_block * 2 + k_loc
                for (i_loc, k_loc), coeff in W_s[s].items():
                    i = i_loc
                    k = b_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return {"U": U, "V": V, "W": W}
