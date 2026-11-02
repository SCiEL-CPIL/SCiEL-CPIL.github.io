import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 98 decomposition using Strassen for 4x4x8
    # Outer Strassen (2x2 blocks) rank 7
    # Inner Strassen (2x2 by 2x4) rank 14 (two 2x2x2 Strassen)
    
    # Define Strassen 2x2 coefficients
    # A_r[i, j], B_r[j, k], C_r[k, i] for i,j,k in {0,1}
    # r in 0..6
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for A (input 1)
    A_coeffs = [
        [[1, 0], [0, 1]],  # M1
        [[0, 0], [1, 1]],  # M2
        [[1, 0], [0, 0]],  # M3
        [[0, 0], [0, 1]],  # M4
        [[1, 1], [0, 0]],  # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]]  # M7
    ]
    
    # Coefficients for B (input 2)
    B_coeffs = [
        [[1, 1], [0, 0]],  # M1
        [[1, 0], [0, 0]],  # M2
        [[0, 1], [0, -1]], # M3
        [[-1, 0], [1, 0]], # M4
        [[0, 0], [0, 1]],  # M5
        [[1, 1], [0, 0]],  # M6
        [[0, 0], [1, 1]]   # M7
    ]
    
    # Coefficients for C (output)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_coeffs = [
        [[1, 0], [0, 0]],  # M1 contribution to C00
        [[0, 0], [0, 0]],  # M2
        [[0, 0], [0, 0]],  # M3
        [[0, 0], [0, 0]],  # M4
        [[0, 0], [0, 0]],  # M5
        [[0, 0], [0, 0]],  # M6
        [[0, 0], [0, 0]]   # M7
    ]
    # Wait, C_coeffs should be indexed by r, then [k, i]
    # C_coeffs[r][k][i]
    
    # Let's rebuild C_coeffs properly
    # r=0 (M1): C00 += 1, C11 += 1
    # r=1 (M2): C10 += 1, C11 += -1
    # r=2 (M3): C01 += 1, C11 += 1
    # r=3 (M4): C00 += 1, C10 += 1
    # r=4 (M5): C00 += -1, C01 += 1
    # r=5 (M6): C11 += 1
    # r=6 (M7): C00 += 1
    
    C_coeffs = [
        [[1, 0], [0, 1]], # r=0: C00=1, C11=1
        [[0, 0], [1, -1]], # r=1: C10=1, C11=-1
        [[0, 1], [0, 1]], # r=2: C01=1, C11=1
        [[0, 1], [1, 0]], # r=3: C00=1, C10=1
        [[-1, 0], [0, 1]], # r=4: C00=-1, C01=1
        [[0, 0], [0, 1]], # r=5: C11=1
        [[1, 0], [0, 0]]  # r=6: C00=1
    ]

    # Inner decomposition: 2x2 by 2x4
    # This is two 2x2x2 Strassen decompositions.
    # For col block 0 (k=0,1), use Strassen r=0..6
    # For col block 1 (k=2,3), use Strassen r=0..6
    
    # Outer decomposition: 4x4 by 4x8
    # Blocks: A is 2x2 blocks of 2x2. B is 2x4 blocks of 2x4.
    # Outer Strassen r_out=0..6
    # Inner Strassen r_in=0..6 for col block 0, r_in=0..6 for col block 1
    
    # Total rank = 7 (outer) * 2 (col blocks) * 7 (inner) = 98
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for r_out in range(7):
        # Outer coefficients
        # A_outer[r_out][i_block][j_block]
        # B_outer[r_out][j_block][k_block]
        # C_outer[r_out][k_block][i_block]
        
        # We need to map outer block indices to local indices
        # A_block[i_block, j_block] is a 2x2 matrix.
        # Local indices i_l, j_l in {0,1}
        # Global i = i_block * 2 + i_l
        # Global j = j_block * 2 + j_l
        
        # B_block[j_block, k_block] is a 2x4 matrix.
        # Local indices j_l, k_l in {0,1} for the 2x2 part?
        # B is 4x8. Blocks are 2x4.
        # j_block in {0,1}. k_block in {0,1} (for the two 4-column groups).
        # Inside B_block, we have 2 rows and 4 columns.
        # We split the 4 columns into two 2-column groups.
        # So B_block[j_block, k_block] is split into B_block[j_block, k_block, sub_k] where sub_k in {0,1}.
        # Global k = k_block * 4 + sub_k * 2 + k_l.
        
        # Let's iterate over inner sub-steps
        for sub_k in range(2): # 0 for first 2 cols, 1 for last 2 cols
            for r_in in range(7):
                # This is the (r_out, sub_k, r_in) term
                # U coefficient: A_outer[r_out][i_block][j_block] * A_inner[r_in][i_l][j_l]
                # V coefficient: B_outer[r_out][j_block][k_block] * B_inner[r_in][j_l][k_l]
                # W coefficient: C_outer[r_out][k_block][i_block] * C_inner[r_in][k_l][i_l]
                
                # We need the outer coefficients for A, B, C
                # A_outer is 2x2 blocks.
                # Let's define A_outer_coeffs[r_out][i_block][j_block]
                
                # From Strassen 2x2:
                # M1: A00+A11 -> [1,0;0,1]
                # M2: A10+A11 -> [0,0;1,1]
                # M3: A00 -> [1,0;0,0]
                # M4: A11 -> [0,0;0,1]
                # M5: A00+A01 -> [1,1;0,0]
                # M6: A10-A00 -> [-1,0;1,0]
                # M7: A01-A11 -> [0,1;0,-1]
                
                # B_outer is 2x2 blocks (of 2x4).
                # M1: B00+B11 -> [1,1;0,0]
                # M2: B00 -> [1,0;0,0]
                # M3: B01-B11 -> [0,1;0,-1]
                # M4: B10-B00 -> [-1,0;1,0]
                # M5: B11 -> [0,0;0,1]
                # M6: B00+B01 -> [1,1;0,0]
                # M7: B10+B11 -> [0,0;1,1]
                
                # C_outer is 2x2 blocks (of 2x4).
                # M1: C00+C11 -> [1,0;0,1]
                # M2: C10-C11 -> [0,0;1,-1]
                # M3: C01+C11 -> [0,1;0,1]
                # M4: C00+C10 -> [0,1;1,0]
                # M5: -C00+C01 -> [-1,0;0,1]
                # M6: C11 -> [0,0;0,1]
                # M7: C00 -> [1,0;0,0]
                
                A_outer = [
                    [[1, 0], [0, 1]],
                    [[0, 0], [1, 1]],
                    [[1, 0], [0, 0]],
                    [[0, 0], [0, 1]],
                    [[1, 1], [0, 0]],
                    [[-1, 0], [1, 0]],
                    [[0, 1], [0, -1]]
                ]
                B_outer = [
                    [[1, 1], [0, 0]],
                    [[1, 0], [0, 0]],
                    [[0, 1], [0, -1]],
                    [[-1, 0], [1, 0]],
                    [[0, 0], [0, 1]],
                    [[1, 1], [0, 0]],
                    [[0, 0], [1, 1]]
                ]
                C_outer = [
                    [[1, 0], [0, 1]],
                    [[0, 0], [1, -1]],
                    [[0, 1], [0, 1]],
                    [[0, 1], [1, 0]],
                    [[-1, 0], [0, 1]],
                    [[0, 0], [0, 1]],
                    [[1, 0], [0, 0]]
                ]
                
                # Inner coefficients (same as Strassen 2x2)
                A_inner = A_coeffs
                B_inner = B_coeffs
                C_inner = C_coeffs
                
                # Fill U, V, W
                for i_block in range(2):
                    for j_block in range(2):
                        a_out_val = A_outer[r_out][i_block][j_block]
                        if a_out_val == 0: continue
                        
                        for i_l in range(2):
                            for j_l in range(2):
                                a_in_val = A_inner[r_in][i_l][j_l]
                                if a_in_val == 0: continue
                                
                                val = a_out_val * a_in_val
                                i = i_block * 2 + i_l
                                j = j_block * 2 + j_l
                                U[i * m + j][col] += val
                                
                for j_block in range(2):
                    for k_block in range(2):
                        b_out_val = B_outer[r_out][j_block][k_block]
                        if b_out_val == 0: continue
                        
                        for j_l in range(2):
                            for k_l in range(2):
                                b_in_val = B_inner[r_in][j_l][k_l]
                                if b_in_val == 0: continue
                                
                                val = b_out_val * b_in_val
                                j = j_block * 2 + j_l
                                # k = k_block * 4 + sub_k * 2 + k_l
                                k = k_block * 4 + sub_k * 2 + k_l
                                V[j * p + k][col] += val
                                
                for k_block in range(2):
                    for i_block in range(2):
                        c_out_val = C_outer[r_out][k_block][i_block]
                        if c_out_val == 0: continue
                        
                        for k_l in range(2):
                            for i_l in range(2):
                                c_in_val = C_inner[r_in][k_l][i_l]
                                if c_in_val == 0: continue
                                
                                val = c_out_val * c_in_val
                                i = i_block * 2 + i_l
                                k = k_block * 4 + sub_k * 2 + k_l
                                W[k * n + i][col] += val
                                
                col += 1
                
    return {"U": U, "V": V, "W": W}

