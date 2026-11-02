"""
Program for Matrix Multiplication 3x4x8 using Recursive Strassen (Schoolbook Base)
Target Rank: ~56 (vs 96 schoolbook)
"""
import numpy as np

DIMENSIONS = [3, 4, 8]

def _build_2x2x2_schoolbook():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using schoolbook.
    U shape: (4, 4)
    V shape: (4, 4)
    W shape: (4, 4)
    """
    rank = 4
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Index mapping for 2x2:
    # A: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i * 2 + j][col] = 1
                V[j * 2 + k][col] = 1
                W[k * 2 + i][col] = 1 # Note: W index is k*n + i for n=2? 
                # Task says W[k*n+i, r]. For 2x2x2, n=2, p=2.
                # W shape is p*n = 4.
                # Index is k*2 + i.
                col += 1
                
    return U, V, W

def _build_4x4x4_strassen():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using Strassen algorithm
    with schoolbook base case for 2x2x2 blocks.
    Rank = 7 * 4 = 28.
    U shape: (16, 28)
    V shape: (16, 28)
    W shape: (16, 28)
    """
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    U_22, V_22, W_22 = _build_2x2x2_schoolbook()
    
    # Strassen Formulas
    # We define the linear combinations for U (A) and V (B)
    # Each formula k uses a specific combination of 2x2 blocks.
    # We need to map the 2x2 block indices to the 4x4 indices.
    # Block 00: rows 0-1, cols 0-1
    # Block 01: rows 0-1, cols 2-3
    # Block 10: rows 2-3, cols 0-1
    # Block 11: rows 2-3, cols 2-3
    
    def get_block_indices(block_id):
        r, c = divmod(block_id, 2)
        r_start = r * 2
        c_start = c * 2
        return [r_start, r_start + 1, c_start, c_start + 1]

    # Strassen products (A_block, B_block)
    # M1: (A00 - A11) * (B01 + B11)
    # M2: (A00 + A11) * (B11)
    # M3: (A00) * (B00 - B11)
    # M4: (A11) * (B00 + B10)
    # M5: (A01 + A11) * (B00)
    # M6: (A00 + A01) * (B11 - B01)
    # M7: (A10 - A00) * (B00 + B10)
    
    # We will add the factors for each M_k.
    # The U and V matrices for M_k are formed by combining the rows of U_22 and V_22.
    # The W matrix is formed by combining the rows of W_22.
    
    for k in range(7):
        offset = k * 4
        # Define the linear combination coefficients for A blocks (4 blocks)
        # and B blocks (4 blocks).
        
        if k == 0: # M1
            A_coeffs = {0: 1, 11: -1}
            B_coeffs = {1: 1, 3: 1}
        elif k == 1: # M2
            A_coeffs = {0: 1, 3: 1}
            B_coeffs = {3: 1}
        elif k == 2: # M3
            A_coeffs = {0: 1}
            B_coeffs = {0: 1, 3: -1}
        elif k == 3: # M4
            A_coeffs = {3: 1}
            B_coeffs = {0: 1, 2: 1}
        elif k == 4: # M5
            A_coeffs = {1: 1, 3: 1}
            B_coeffs = {0: 1}
        elif k == 5: # M6
            A_coeffs = {0: 1, 1: 1}
            B_coeffs = {3: 1, 1: -1}
        else: # M7
            A_coeffs = {2: 1, 0: -1}
            B_coeffs = {0: 1, 2: 1}
            
        # Populate U, V, W for this block
        # U_22 is 4x4. The rows correspond to A_{00}, A_{01}, A_{10}, A_{11}.
        # V_22 is 4x4. The rows correspond to B_{00}, B_{01}, B_{10}, B_{11}.
        # W_22 is 4x4. The rows correspond to C_{00}, C_{01}, C_{10}, C_{11}.
        
        for local_a_row in range(4):
            for local_b_row in range(4):
                # local_a_row corresponds to A_block r_a, c_a
                r_a, c_a = divmod(local_a_row, 2)
                a_block_id = r_a * 2 + c_a
                coeff_a = A_coeffs.get(a_block_id, 0)
                
                # local_b_row corresponds to B_block r_b, c_b
                r_b, c_b = divmod(local_b_row, 2)
                b_block_id = r_b * 2 + c_b
                coeff_b = B_coeffs.get(b_block_id, 0)
                
                # The scalar for this specific bilinear term in the Strassen product
                # is coeff_a * coeff_b
                # Wait, the Strassen product is (Sum coeff_a A) * (Sum coeff_b B).
                # This expands to Sum(coeff_a coeff_b A B).
                # So yes, the coefficient is coeff_a * coeff_b.
                
                val = coeff_a * coeff_b
                if val == 0:
                    continue
                    
                # Add to U
                # The global row index in U for A_block a_block_id, local row local_a_row
                # A_block a_block_id has global rows: 2*(a_block_id//2), 2*(a_block_id//2)+1
                # A_block a_block_id has global cols: 2*(a_block_id%2), 2*(a_block_id%2)+1
                # Local row local_a_row corresponds to local r_a, c_a.
                # Global row = 2*r_a + r_a? No.
                # Global Row Index = 2 * (block_row) + local_row
                # block_row = a_block_id // 2
                # local_row = local_a_row // 2
                g_row_a = 2 * (a_block_id // 2) + (local_a_row // 2)
                g_col_a = 2 * (a_block_id % 2) + (local_a_row % 2)
                U_idx = g_row_a * 4 + g_col_a
                
                U[U_idx, offset:offset+4] += val * U_22[local_a_row, :]
                
                # Add to V
                g_row_b = 2 * (b_block_id // 2) + (local_b_row // 2)
                g_col_b = 2 * (b_block_id % 2) + (local_b_row % 2)
                V_idx = g_row_b * 4 + g_col_b
                
                V[V_idx, offset:offset+4] += val * V_22[local_b_row, :]
                
                # Add to W
                # W_22 rows correspond to C blocks.
                # C_block c_block_id has global rows: 2*(c_block_id//2), ...
                # C_block c_block_id has global cols: 2*(c_block_id%2), ...
                # local_w_row in W_22 corresponds to c_block_id?
                # W_22 is 4x4. Row 0: C00, Row 1: C01, Row 2: C10, Row 3: C11.
                # So for each local_w_row in 0..3:
                for local_w_row in range(4):
                    c_block_id = local_w_row
                    # Global W index
                    g_row_c = 2 * (c_block_id // 2) + (local_w_row // 2) # Wait, W_22 row is c_block_id?
                    # W_22[local_w_row, col] is the coefficient for C_{c_block_id, col_in_block}
                    # c_block_id = local_w_row
                    # local col = col_in_block
                    g_col_c = 2 * (c_block_id % 2) + (local_w_row % 2) # No, local_w_row is the row index of W_22.
                    # W_22 is defined such that W_22[r, c] maps to C_block r, local col c?
                    # In _build_2x2x2_schoolbook, W[k*n+i, r].
                    # For 2x2, n=2. W index k*2+i.
                    # k is row of C, i is col of C? No, i is row of A, k is col of B?
                    # Task: W[k*n+i, r].
                    # C[i,k] = sum A[i,j] B[j,k].
                    # So W index depends on i (row of C) and k (col of C).
                    # In my _build_2x2x2_schoolbook:
                    # W[k * 2 + i][col] = 1.
                    # So W row index is k*2 + i.
                    # k is 0..1 (col of C), i is 0..1 (row of C).
                    # So W_22 row index r_w corresponds to k = r_w // 2, i = r_w % 2.
                    # So c_block_id is NOT simply local_w_row.
                    
                    # Let's re-map W_22 rows.
                    # r_w = k*2 + i.
                    # k = r_w // 2.
                    # i = r_w % 2.
                    # C global row = i.
                    # C global col = k * 2 + c_in_block? No.
                    # The Strassen product M_k is a 2x2 matrix.
                    # Its (i,j) entry contributes to C at global position (block_row*2 + i, block_col*2 + j).
                    # In the 2x2x2 schoolbook, C[i,k] is the entry.
                    # So W_22 row index r_w = k*2 + i.
                    # Here k is the col index of the 2x2 result, i is the row index.
                    # So for the 2x2 block M_k:
                    # Result row i, col k.
                    # Global row = block_row * 2 + i.
                    # Global col = block_col * 2 + k.
                    # But which block?
                    # The Strassen formulas M1..M7 are 2x2 matrices.
                    # They contribute to specific C blocks?
                    # No, M1..M7 are the *intermediate* 2x2 matrices.
                    # The final C is assembled from M1..M7.
                    
                    # WAIT. My Strassen implementation above is for the *values* of M1..M7.
                    # M1 is a 2x2 matrix.
                    # C is assembled as linear combinations of M1..M7.
                    # C_00 = M2_00 + M3_00 - M4_00 + M7_00
                    # C_01 = M1_00 + M4_00 ... wait, M1 is 2x2, so M1_00 is the top-left of M1.
                    
                    # This means I cannot just add W_22 directly.
                    # I must apply the Strassen reconstruction formulas to the W columns.
                    
                    # Let's restart the W assembly logic.
                    # We have 7 sets of 4 W columns (one for each M_k).
                    # Let W_k be the 4x4 matrix for M_k.
                    # W_k[r, c] is the coefficient for C_{i,k} in the 2x2 block M_k?
                    # No, W_k is the factor matrix for the 2x2x2 multiplication that computes M_k.
                    # So W_k[r, c] tells us how the output of M_k is formed.
                    # Actually, W_k[r, c] is the coefficient for the output element C_{i,k} of the 2x2 matrix M_k.
                    # The output of M_k is a 2x2 matrix. Let's call it M_k_mat.
                    # M_k_mat[i, k] = sum ...
                    # The W matrix in the tensor decomposition maps the bilinear products to the output.
                    # So W_k[r, c] = 1 if the output element is M_k_mat[i, k].
                    
                    # The final C is:
                    # C_00 = M2_mat[0,0] + M3_mat[0,0] - M4_mat[0,0] + M7_mat[0,0]
                    # C_01 = M1_mat[0,0] + M4_mat[0,0] ... No.
                    # C_01 = M1_mat[0,0] + M4_mat[0,0]?
                    # Standard Strassen:
                    # C_00 = M2_00 + M3_00 - M4_00 + M7_00
                    # C_01 = M1_00 + M4_00
                    # C_10 = M2_10 + M5_10
                    # C_11 = M1_11 - M2_11 + M3_11 + M6_11
                    
                    # Note: M_k are 2x2 matrices.
                    # So we need to combine the W columns from different M_k's.
                    
                    # This part is complex. I will hardcode the W reconstruction.
                    pass # Placeholder to avoid syntax error in thought process

# Since the W reconstruction is complex, I will use a simpler, guaranteed approach:
# Just use Schoolbook for the whole thing but optimized? No, 96 is the max.
# I will use the Rank-28 4x4x4 Strassen but implement the W assembly carefully.

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the 4x4x8 decomposition.
    # Rank 56.
    rank = 56
    U = np.zeros((16, rank)) # 4x4 A
    V = np.zeros((32, rank)) # 4x8 B
    W = np.zeros((32, rank)) # 4x8 C
    
    # Build 4x4x4 Strassen factors
    # We need to generate the U, V, W for one 4x4x4 block.
    
    def build_4x4x4_block():
        rank_blk = 28
        U_blk = np.zeros((16, rank_blk))
        V_blk = np.zeros((16, rank_blk))
        W_blk = np.zeros((16, rank_blk))
        
        U_22, V_22, W_22 = _build_2x2x2_schoolbook()
        
        # Strassen U/V combinations
        for k in range(7):
            offset = k * 4
            if k == 0:
                A_c = {0: 1, 3: -1}
                B_c = {1: 1, 3: 1}
            elif k == 1:
                A_c = {0: 1, 3: 1}
                B_c = {3: 1}
            elif k == 2:
                A_c = {0: 1}
                B_c = {0: 1, 3: -1}
            elif k == 3:
                A_c = {3: 1}
                B_c = {0: 1, 2: 1}
            elif k == 4:
                A_c = {1: 1, 3: 1}
                B_c = {0: 1}
            elif k == 5:
                A_c = {0: 1, 1: 1}
                B_c = {3: 1, 1: -1}
            else:
                A_c = {2: 1, 0: -1}
                B_c = {0: 1, 2: 1}
                
            for la in range(4):
                ra, ca = divmod(la, 2)
                bid_a = ra * 2 + ca
                val_a = A_c.get(bid_a, 0)
                if val_a == 0: continue
                
                gr_a = 2 * (bid_a // 2) + (la // 2)
                gc_a = 2 * (bid_a % 2) + (la % 2)
                U_idx = gr_a * 4 + gc_a
                U_blk[U_idx, offset:offset+4] += val_a * U_22[la, :]
                
            for lb in range(4):
                rb, cb = divmod(lb, 2)
                bid_b = rb * 2 + cb
                val_b = B_c.get(bid_b, 0)
                if val_b == 0: continue
                
                gr_b = 2 * (bid_b // 2) + (lb // 2)
                gc_b = 2 * (bid_b % 2) + (lb % 2)
                V_idx = gr_b * 4 + gc_b
                V_blk[V_idx, offset:offset+4] += val_b * V_22[lb, :]
                
        # W Reconstruction
        # C_00 = M2[0,0] + M3[0,0] - M4[0,0] + M7[0,0]
        # C_01 = M1[0,0] + M4[0,0]
        # C_10 = M2[1,0] + M5[1,0]
        # C_11 = M1[1,1] - M2[1,1] + M3[1,1] + M6[1,1]
        
        # M_k is 2x2. M_k[i,j] corresponds to W_22 row index j*2+i?
        # In _build_2x2x2_schoolbook, W[k*n+i, r]. n=2.
        # W index = k*2 + i.
        # k is col, i is row.
        # So M_k[i,j] is in W_22 row j*2 + i.
        
        # C_00 (i=0, k=0) -> W_22 row 0.
        # C_01 (i=0, k=1) -> W_22 row 2.
        # C_10 (i=1, k=0) -> W_22 row 1.
        # C_11 (i=1, k=1) -> W_22 row 3.
        
        # Combine:
        # C_00: M2(row 0) + M3(row 0) - M4(row 0) + M7(row 0)
        # C_01: M1(row 2) + M4(row 2)
        # C_10: M2(row 1) + M5(row 1)
        # C_11: M1(row 3) - M2(row 3) + M3(row 3) + M6(row 3)
        
        # M_k offset is k*4.
        
        W_blk[0:4, :] = W_22[0, 4:8] * 1 + W_22[0, 8:12] * 1 - W_22[0, 12:16] * 1 + W_22[0, 24:28] * 1
        W_blk[4:8, :] = W_22[2, 0:4] * 1 + W_22[2, 12:16] * 1
        W_blk[8:12, :] = W_22[1, 4:8] * 1 + W_22[1, 16:20] * 1
        W_blk[12:16, :] = W_22[3, 0:4] * 1 - W_22[3, 4:8] * 1 + W_22[3, 8:12] * 1 + W_22[3, 20:24] * 1
        
        return U_blk, V_blk, W_blk
    
    U_blk, V_blk, W_blk = build_4x4x4_block()
    
    # Combine two 4x4x4 blocks for 4x4x8
    U[:, 0:28] = U_blk
    U[:, 28:56] = U_blk
    
    V[0:16, 0:28] = V_blk
    V[16:32, 28:56] = V_blk
    
    W[0:16, 0:28] = W_blk
    W[16:32, 28:56] = W_blk
    
    # Truncate to 3x4x8
    # U: first 3 rows (0,1,2)
    U_new = U[0:3, :]
    # V: all rows
    V_new = V
    # W: first 3 rows (0,1,2)
    W_new = W[0:3, :]
    
    return {
        "U": U_new.tolist(),
        "V": V_new.tolist(),
        "W": W_new.tolist()
    }

