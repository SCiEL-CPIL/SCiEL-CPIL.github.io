import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Schoolbook solution as fallback
    rank_sb = n * m * p  # 96
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Attempt to construct the Rank 84 solution based on Block Strassen
    # Strategy: Decompose into 12 independent 2x2x2 multiplications (6 output blocks, each sum of 2 products)
    # Each 2x2x2 product uses Strassen (rank 7). Total rank 12 * 7 = 84.
    
    try:
        rank_84 = 84
        U_84 = [[0.0] * rank_84 for _ in range(n * m)]
        V_84 = [[0.0] * rank_84 for _ in range(m * p)]
        W_84 = [[0.0] * rank_84 for _ in range(p * n)]
        
        # Strassen coefficients for 2x2 matrix multiplication
        # Inputs: A (2x2), B (2x2)
        # A indices: a00, a01, a10, a11
        # B indices: b00, b01, b10, b11
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We have 6 output blocks C_ij where i in {0,1}, j in {0,1,2}
        # C_ij = A_i0 B_0j + A_i1 B_1j
        # Let's iterate over output blocks and the two terms in the sum
        
        col = 0
        for i_block in range(2): # i index of 2x2 block in A and C
            for j_block in range(3): # j index of 2x2 block in B and C
                
                # Term 1: A_i0 * B_0j
                # A_i0 is rows 2*i_block to 2*i_block+1, cols 0 to 1
                # B_0j is rows 0 to 1, cols 2*j_block to 2*j_block+1
                # C_ij is rows 2*i_block to 2*i_block+1, cols 2*j_block to 2*j_block+1
                
                a_rows = [2*i_block, 2*i_block+1]
                a_cols = [0, 1]
                b_rows = [0, 1]
                b_cols = [2*j_block, 2*j_block+1]
                c_rows = [2*i_block, 2*i_block+1]
                c_cols = [2*j_block, 2*j_block+1]
                
                # Apply Strassen to this 2x2 product
                # M1
                U_84[a_rows[0]][col] += 1; U_84[a_rows[1]][col] += 1 # a00+a11
                V_84[b_rows[0]][col] += 1; V_84[b_rows[1]][col] += 1 # b00+b11
                W_84[c_rows[0]][col] += 1; W_84[c_rows[1]][col] += 1 # C00, C11
                col += 1
                
                # M2
                U_84[a_rows[1]][col] += 1; U_84[a_rows[1]][col] += 1 # a10+a11 (a10 is row1 col0, a11 is row1 col1)
                # Wait, indices:
                # A_i0:
                # Row 0: A[2i, 0], A[2i, 1]  -> a00, a01
                # Row 1: A[2i+1, 0], A[2i+1, 1] -> a10, a11
                # So a10 is U[2i+1, 0], a11 is U[2i+1, 1].
                # M2 = (a10+a11) * b00
                U_84[2*i_block+1][col] += 1 # a10
                U_84[2*i_block+1][col] += 1 # a11
                V_84[0][col] += 1 # b00 (row 0, col 0 of B_0j which is global col 2j)
                W_84[c_rows[1]][col] += 1 # C10
                col += 1
                
                # M3
                # M3 = a00 * (b10 - b11)
                U_84[2*i_block][col] += 1 # a00
                V_84[1][col] += 1 # b10 (row 1, col 0 of B_0j -> global col 2j)
                V_84[1][col] -= 1 # b11 (row 1, col 1 of B_0j -> global col 2j+1)
                W_84[c_rows[0]][col] += 1 # C01
                W_84[c_rows[1]][col] += 1 # C11
                col += 1
                
                # M4
                # M4 = a11 * (b01 - b00)
                U_84[2*i_block+1][col] += 1 # a11
                V_84[0][col] += 1 # b01 (row 0, col 1 of B_0j -> global col 2j+1)
                V_84[0][col] -= 1 # b00 (row 0, col 0 of B_0j -> global col 2j)
                W_84[c_rows[0]][col] += 1 # C00
                W_84[c_rows[1]][col] += 1 # C10
                col += 1
                
                # M5
                # M5 = (a00 + a01) * b11
                U_84[2*i_block][col] += 1 # a00
                U_84[2*i_block][col] += 1 # a01
                V_84[1][col] += 1 # b11 (row 1, col 1 of B_0j -> global col 2j+1)
                W_84[c_rows[0]][col] -= 1 # C00
                W_84[c_rows[1]][col] += 1 # C01
                col += 1
                
                # M6
                # M6 = (a10 - a00) * (b00 + b01)
                U_84[2*i_block+1][col] += 1 # a10
                U_84[2*i_block][col] -= 1 # a00
                V_84[0][col] += 1 # b00
                V_84[0][col] += 1 # b01
                W_84[c_rows[1]][col] += 1 # C11
                col += 1
                
                # M7
                # M7 = (a01 - a11) * (b10 + b11)
                U_84[2*i_block][col] += 1 # a01
                U_84[2*i_block+1][col] -= 1 # a11
                V_84[1][col] += 1 # b10
                V_84[1][col] += 1 # b11
                W_84[c_rows[0]][col] += 1 # C00
                W_84[c_rows[1]][col] += 1 # C11
                col += 1
                
                # Term 2: A_i1 * B_1j
                # A_i1 is rows 2*i_block to 2*i_block+1, cols 2 to 3
                # B_1j is rows 2 to 3, cols 2*j_block to 2*j_block+1
                
                a_cols_t2 = [2, 3]
                b_rows_t2 = [2, 3]
                
                # Apply Strassen to this 2x2 product
                # M1'
                U_84[a_rows[0]][col] += 1; U_84[a_rows[1]][col] += 1 # a00+a11 of A_i1
                V_84[b_rows_t2[0]][col] += 1; V_84[b_rows_t2[1]][col] += 1 # b00+b11 of B_1j
                W_84[c_rows[0]][col] += 1; W_84[c_rows[1]][col] += 1 # C00, C11
                col += 1
                
                # M2'
                # M2' = (a10+a11) * b00
                U_84[2*i_block+1][col] += 1 # a10 of A_i1 (global row 2i+1, col 2)
                U_84[2*i_block+1][col] += 1 # a11 of A_i1 (global row 2i+1, col 3)
                V_84[2][col] += 1 # b00 of B_1j (global row 2, col 2j)
                W_84[c_rows[1]][col] += 1 # C10
                col += 1
                
                # M3'
                # M3' = a00 * (b10 - b11)
                U_84[2*i_block][col] += 1 # a00 of A_i1 (global row 2i, col 2)
                V_84[3][col] += 1 # b10 of B_1j (global row 3, col 2j)
                V_84[3][col] -= 1 # b11 of B_1j (global row 3, col 2j+1)
                W_84[c_rows[0]][col] += 1 # C01
                W_84[c_rows[1]][col] += 1 # C11
                col += 1
                
                # M4'
                # M4' = a11 * (b01 - b00)
                U_84[2*i_block+1][col] += 1 # a11 of A_i1 (global row 2i+1, col 3)
                V_84[2][col] += 1 # b01 of B_1j (global row 2, col 2j+1)
                V_84[2][col] -= 1 # b00 of B_1j (global row 2, col 2j)
                W_84[c_rows[0]][col] += 1 # C00
                W_84[c_rows[1]][col] += 1 # C10
                col += 1
                
                # M5'
                # M5' = (a00 + a01) * b11
                U_84[2*i_block][col] += 1 # a00 of A_i1
                U_84[2*i_block][col] += 1 # a01 of A_i1
                V_84[3][col] += 1 # b11 of B_1j (global row 3, col 2j+1)
                W_84[c_rows[0]][col] -= 1 # C00
                W_84[c_rows[1]][col] += 1 # C01
                col += 1
                
                # M6'
                # M6' = (a10 - a00) * (b00 + b01)
                U_84[2*i_block+1][col] += 1 # a10 of A_i1
                U_84[2*i_block][col] -= 1 # a00 of A_i1
                V_84[2][col] += 1 # b00 of B_1j
                V_84[2][col] += 1 # b01 of B_1j
                W_84[c_rows[1]][col] += 1 # C11
                col += 1
                
                # M7'
                # M7' = (a01 - a11) * (b10 + b11)
                U_84[2*i_block][col] += 1 # a01 of A_i1
                U_84[2*i_block+1][col] -= 1 # a11 of A_i1
                V_84[3][col] += 1 # b10 of B_1j
                V_84[3][col] += 1 # b11 of B_1j
                W_84[c_rows[0]][col] += 1 # C00
                W_84[c_rows[1]][col] += 1 # C11
                col += 1
        
        # Verify the rank 84 solution
        if col == 84:
             U_np = np.array(U_84)
             V_np = np.array(V_84)
             W_np = np.array(W_84)
             
             valid = True
             for i in range(n):
                 for k in range(p):
                     for j in range(m):
                         for l in range(m):
                             val = 0.0
                             for r in range(84):
                                 val += W_np[k*n+i, r] * U_np[i*m+j, r] * V_np[l*p+k, r]
                             expected = 1.0 if j == l else 0.0
                             if abs(val - expected) > 1e-8:
                                 valid = False
                                 break
                         if not valid: break
                     if not valid: break
                 if not valid: break
             
             if valid:
                 best_solution = {
                     "U": [[float(v) for v in row] for row in U_84],
                     "V": [[float(v) for v in row] for row in V_84],
                     "W": [[float(v) for v in row] for row in W_84]
                 }
                 best_rank = 84
    except Exception as e:
        # Fallback to schoolbook if construction fails
        pass

    return best_solution
