"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of 4x4x9 matrix multiplication using Strassen for 4x4x4 blocks.
    # Strategy: Treat the 9 columns as three 3-column blocks? No, that's messy.
    # Better strategy: Treat the 9 columns as a 4x4 block (cols 0-3), a 4x4 block (cols 4-8), 
    # and handle the overlap? No.
    # Actually, 4x4x9 can be decomposed by padding to 4x4x12 (3 blocks of 4x4x4) and optimizing?
    # Or simply use Strassen (rank 49) for the first 4 columns, Strassen for next 4 columns, 
    # and schoolbook for the last 1 column?
    # Rank = 49 + 49 + 4 = 102.
    # This is strictly better than 144.
    
    n, m, p = DIMENSIONS
    # p is 9. We split p into 4, 4, 1.
    p_blocks = [4, 4, 1]
    
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    def get_strassen_2x2():
        # Returns U, V, W for 2x2x2 multiplication (rank 7)
        # Indices:
        # U rows: (a,b,c,d) -> 0,1,2,3.  U[i, k] is coeff of A[i] in linear form k.
        # V rows: (x,y,z,w) -> 0,1,2,3.
        # W rows: (C11,C12,C21,C22) -> 0,1,2,3.
        # Strassen formulas:
        # M0 = (a+d)(x+w)
        # M1 = (c+d)x
        # M2 = a(y-w)
        # M3 = d(z-x)
        # M4 = (a+b)w
        # M5 = (c-a)(x+y)
        # M6 = (b-d)(z+w)
        
        # Linear forms for A (rows of U):
        # k=0: a+d -> [1,0,0,1]
        # k=1: c+d -> [0,0,1,1]
        # k=2: a   -> [1,0,0,0]
        # k=3: d   -> [0,0,0,1]
        # k=4: a+b -> [1,1,0,0]
        # k=5: c-a -> [-1,0,1,0]
        # k=6: b-d -> [0,1,0,-1]
        
        U = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        
        # Linear forms for B (rows of V):
        # k=0: x+w -> [1,0,0,1]
        # k=1: x   -> [1,0,0,0]
        # k=2: y-w -> [0,1,0,-1]
        # k=3: z-x -> [-1,0,1,0]
        # k=4: w   -> [0,0,0,1]
        # k=5: x+y -> [1,1,0,0]
        # k=6: z+w -> [0,0,1,1]
        
        V = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        
        # Output combination (rows of W):
        # C11 = M0 + M3 - M4 + M6 -> [1,0,0,1,-1,0,1]
        # C12 = M2 + M4           -> [0,0,1,0,1,0,0]
        # C21 = M1 + M3           -> [0,1,0,1,0,0,0]
        # C22 = M0 - M1 + M2 + M5 -> [1,-1,1,0,0,1,0]
        
        W = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        return U, V, W
    
    def get_strassen_4x4():
        # Returns U, V, W for 4x4x4 multiplication (rank 49)
        # Constructed from 7 blocks of 2x2x2 Strassen.
        # U shape: (16, 49)
        # V shape: (16, 49)
        # W shape: (16, 49)
        
        sU, sV, sW = get_strassen_2x2()
        
        # The 7 blocks correspond to the 7 Strassen terms applied to 2x2 submatrices.
        # Let A be 4x4. Submatrices A11, A12, A21, A22 (each 2x2).
        # B 4x4. Submatrices B11, B12, B21, B22.
        # C = A*B.
        
        # We use the 4x4 Strassen algorithm which itself uses 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication uses rank 7. Total rank 49.
        
        # The 7 high-level Strassen terms for 4x4:
        # M0 = (A11+A22)(B11+B22)
        # M1 = (A21+A22)B11
        # M2 = A11(B12-B22)
        # M3 = A22(B21-B11)
        # M4 = (A11+A12)B22
        # M5 = (A21-A11)(B11+B12)
        # M6 = (A12-A22)(B21+B22)
        
        # We need to map the linear forms.
        # Let's define the 49 rank-1 tensors.
        # For each high-level term Mk, we use a 2x2 Strassen decomposition.
        # The 2x2 Strassen decomposes X*Y into 7 scalar multiplications.
        
        # U matrix rows correspond to entries of A (16 entries).
        # V matrix rows correspond to entries of B (16 entries).
        # W matrix rows correspond to entries of C (16 entries).
        
        # Let's build U, V, W as lists of length 49 (one column per rank-1 term).
        
        # Helper to get index in 4x4 matrix (row-major):
        def idx4(r, c): return r * 4 + c
        # Helper to get index in 2x2 submatrix relative to offset (r_off, c_off):
        # We will construct the full 16x49 matrices directly.
        
        U_full = [[0]*49 for _ in range(16)]
        V_full = [[0]*49 for _ in range(16)]
        W_full = [[0]*49 for _ in range(16)]
        
        # Define the 7 high-level blocks.
        # Each block k has a start column in the 49-dim space.
        
        # Block 0: M0 = (A11+A22)(B11+B22)
        # Block 1: M1 = (A21+A22)B11
        # Block 2: M2 = A11(B12-B22)
        # Block 3: M3 = A22(B21-B11)
        # Block 4: M4 = (A11+A12)B22
        # Block 5: M5 = (A21-A11)(B11+B12)
        # Block 6: M6 = (A12-A22)(B21+B22)
        
        # We need to express the linear forms for the inputs to each 2x2 block.
        # And the output combination for each 2x2 block.
        
        # Let's precompute the 2x2 Strassen U,V,W matrices as numpy arrays for easier manipulation
        import numpy as np
        sU_np = np.array(sU)
        sV_np = np.array(sV)
        sW_np = np.array(sW)
        
        # For each high-level block k (0..6):
        # We need to define:
        # 1. How the 4 inputs of the first 2x2 matrix are formed from A.
        # 2. How the 4 inputs of the second 2x2 matrix are formed from B.
        # 3. How the 4 outputs of the 2x2 product are added to C.
        
        # Let's define mappings for each block.
        # A is 4x4.
        # A11: rows 0-1, cols 0-1.
        # A12: rows 0-1, cols 2-3.
        # A21: rows 2-3, cols 0-1.
        # A22: rows 2-3, cols 2-3.
        
        # B is 4x4.
        # B11, B12, B21, B22 similarly.
        
        # C is 4x4.
        # C11, C12, C21, C22.
        
        # Block k=0: X = A11+A22, Y = B11+B22
        # X is a 2x2 matrix. Its entries are linear combinations of A.
        # X_00 = A_00 + A_22
        # X_01 = A_01 + A_23
        # X_10 = A_10 + A_32
        # X_11 = A_11 + A_33
        
        # We need to expand the 4 input vectors of the 2x2 Strassen (a,b,c,d) into the 16 A entries.
        # The 2x2 Strassen U matrix has rows for a,b,c,d.
        # U_full[:, k*7 + j] will be the coefficient vector for the j-th scalar product in block k.
        
        # For a given block, we have a linear map L_A: R^16 -> R^4 (mapping A to the 2x2 input X)
        # and L_B: R^16 -> R^4 (mapping B to the 2x2 input Y)
        # and L_C: R^4 -> R^16 (mapping the 2x2 output Z to the 4 entries of C)
        
        # Actually, it's easier to just build the columns.
        
        def add_block(col_offset, map_A, map_B, map_C):
            # map_A: 4x16 matrix. map_A[r, i] is coeff of A[i] in X_r.
            # map_B: 4x16 matrix. map_B[r, i] is coeff of B[i] in Y_r.
            # map_C: 16x4 matrix. map_C[i, r] is coeff of Z_r in C[i].
            
            # U columns:
            # For j in 0..6 (scalar products in 2x2 Strassen):
            # The linear form for X is sU_np[:, j] (4x1).
            # The full linear form for A is map_A @ sU_np[:, j] (16x1).
            # So U_full[:, col_offset + j] = map_A @ sU_np[:, j]
            
            # Similarly for V: V_full[:, col_offset + j] = map_B @ sV_np[:, j]
            
            # For W:
            # The output Z is 4x1.
            # The contribution to C is map_C @ Z (16x1).
            # Z is determined by the scalar products m_0..m_6.
            # Z = sW_np @ m (4x1), where m is vector of scalar products.
            # So the coefficient of m_j in C is map_C @ sW_np[:, j].
            # W_full[:, col_offset + j] = map_C @ sW_np[:, j]
            
            for j in range(7):
                u_vec = map_A.dot(sU_np[:, j])
                v_vec = map_B.dot(sV_np[:, j])
                w_vec = map_C.dot(sW_np[:, j])
                
                for i in range(16):
                    U_full[i][col_offset + j] = u_vec[i]
                    V_full[i][col_offset + j] = v_vec[i]
                    W_full[i][col_offset + j] = w_vec[i]
        
        # Define map_A, map_B, map_C for each block.
        
        # Block 0: X = A11 + A22
        # X_00 = A_00 + A_22
        # X_01 = A_01 + A_23
        # X_10 = A_10 + A_32
        # X_11 = A_11 + A_33
        map_A0 = np.zeros((4, 16))
        map_A0[0, 0] = 1; map_A0[0, 10] = 1 # A_00, A_22 (idx 2*4+2=10)
        map_A0[1, 1] = 1; map_A0[1, 11] = 1 # A_01, A_23 (idx 2*4+3=11)
        map_A0[2, 4] = 1; map_A0[2, 14] = 1 # A_10, A_32 (idx 1*4+0=4, 3*4+2=14)
        map_A0[3, 5] = 1; map_A0[3, 15] = 1 # A_11, A_33 (idx 1*4+1=5, 3*4+3=15)
        
        map_B0 = np.zeros((4, 16))
        map_B0[0, 0] = 1; map_B0[0, 10] = 1 # B_00, B_22
        map_B0[1, 1] = 1; map_B0[1, 11] = 1 # B_01, B_23
        map_B0[2, 4] = 1; map_B0[2, 14] = 1 # B_10, B_32
        map_B0[3, 5] = 1; map_B0[3, 15] = 1 # B_11, B_33
        
        # C = Z (identity mapping for C11 block? No, Z is the 2x2 result for C11)
        # Z_00 -> C_00, Z_01 -> C_01, Z_10 -> C_10, Z_11 -> C_11
        map_C0 = np.zeros((16, 4))
        map_C0[0, 0] = 1
        map_C0[1, 1] = 1
        map_C0[4, 2] = 1
        map_C0[5, 3] = 1
        
        add_block(0, map_A0, map_B0, map_C0)
        
        # Block 1: X = A21 + A22, Y = B11
        # X_00 = A_20 + A_22
        # X_01 = A_21 + A_23
        # X_10 = A_30 + A_32
        # X_11 = A_31 + A_33
        map_A1 = np.zeros((4, 16))
        map_A1[0, 8] = 1; map_A1[0, 10] = 1 # A_20, A_22
        map_A1[1, 9] = 1; map_A1[1, 11] = 1 # A_21, A_23
        map_A1[2, 12] = 1; map_A1[2, 14] = 1 # A_30, A_32
        map_A1[3, 13] = 1; map_A1[3, 15] = 1 # A_31, A_33
        
        map_B1 = np.zeros((4, 16))
        map_B1[0, 0] = 1 # B_00
        map_B1[1, 1] = 1 # B_01
        map_B1[2, 4] = 1 # B_10
        map_B1[3, 5] = 1 # B_11
        
        # C21 = Z
        map_C1 = np.zeros((16, 4))
        map_C1[8, 0] = 1
        map_C1[9, 1] = 1
        map_C1[12, 2] = 1
        map_C1[13, 3] = 1
        
        add_block(7, map_A1, map_B1, map_C1)
        
        # Block 2: X = A11, Y = B12 - B22
        # Y_00 = B_02 - B_22
        # Y_01 = B_03 - B_23
        # Y_10 = B_12 - B_32
        # Y_11 = B_13 - B_33
        map_A2 = np.zeros((4, 16))
        map_A2[0, 0] = 1 # A_00
        map_A2[1, 1] = 1 # A_01
        map_A2[2, 4] = 1 # A_10
        map_A2[3, 5] = 1 # A_11
        
        map_B2 = np.zeros((4, 16))
        map_B2[0, 2] = 1; map_B2[0, 10] = -1 # B_02, -B_22
        map_B2[1, 3] = 1; map_B2[1, 11] = -1 # B_03, -B_23
        map_B2[2, 6] = 1; map_B2[2, 14] = -1 # B_12, -B_32
        map_B2[3, 7] = 1; map_B2[3, 15] = -1 # B_13, -B_33
        
        # C12 = Z
        map_C2 = np.zeros((16, 4))
        map_C2[2, 0] = 1
        map_C2[3, 1] = 1
        map_C2[6, 2] = 1
        map_C2[7, 3] = 1
        
        add_block(14, map_A2, map_B2, map_C2)
        
        # Block 3: X = A22, Y = B21 - B11
        # Y_00 = B_20 - B_00
        # Y_01 = B_21 - B_01
        # Y_10 = B_30 - B_10
        # Y_11 = B_31 - B_11
        map_A3 = np.zeros((4, 16))
        map_A3[0, 10] = 1 # A_22
        map_A3[1, 11] = 1 # A_23
        map_A3[2, 14] = 1 # A_32
        map_A3[3, 15] = 1 # A_33
        
        map_B3 = np.zeros((4, 16))
        map_B3[0, 8] = 1; map_B3[0, 0] = -1 # B_20, -B_00
        map_B3[1, 9] = 1; map_B3[1, 1] = -1 # B_21, -B_01
        map_B3[2, 12] = 1; map_B3[2, 4] = -1 # B_30, -B_10
        map_B3[3, 13] = 1; map_B3[3, 5] = -1 # B_31, -B_11
        
        # C22 = Z
        map_C3 = np.zeros((16, 4))
        map_C3[10, 0] = 1
        map_C3[11, 1] = 1
        map_C3[14, 2] = 1
        map_C3[15, 3] = 1
        
        add_block(21, map_A3, map_B3, map_C3)
        
        # Block 4: X = A11 + A12, Y = B22
        # X_00 = A_00 + A_02
        # X_01 = A_01 + A_03
        # X_10 = A_10 + A_12
        # X_11 = A_11 + A_13
        map_A4 = np.zeros((4, 16))
        map_A4[0, 0] = 1; map_A4[0, 2] = 1
        map_A4[1, 1] = 1; map_A4[1, 3] = 1
        map_A4[2, 4] = 1; map_A4[2, 6] = 1
        map_A4[3, 5] = 1; map_A4[3, 7] = 1
        
        map_B4 = np.zeros((4, 16))
        map_B4[0, 10] = 1 # B_22
        map_B4[1, 11] = 1 # B_23
        map_B4[2, 14] = 1 # B_32
        map_B4[3, 15] = 1 # B_33
        
        # C12 += Z
        map_C4 = np.zeros((16, 4))
        map_C4[2, 0] = 1
        map_C4[3, 1] = 1
        map_C4[6, 2] = 1
        map_C4[7, 3] = 1
        
        add_block(28, map_A4, map_B4, map_C4)
        
        # Block 5: X = A21 - A11, Y = B11 + B12
        # X_00 = A_20 - A_00
        # X_01 = A_21 - A_01
        # X_10 = A_30 - A_10
        # X_11 = A_31 - A_11
        map_A5 = np.zeros((4, 16))
        map_A5[0, 8] = 1; map_A5[0, 0] = -1
        map_A5[1, 9] = 1; map_A5[1, 1] = -1
        map_A5[2, 12] = 1; map_A5[2, 4] = -1
        map_A5[3, 13] = 1; map_A5[3, 5] = -1
        
        map_B5 = np.zeros((4, 16))
        map_B5[0, 0] = 1; map_B5[0, 2] = 1
        map_B5[1, 1] = 1; map_B5[1, 3] = 1
        map_B5[2, 4] = 1; map_B5[2, 6] = 1
        map_B5[3, 5] = 1; map_B5[3, 7] = 1
        
        # C22 += Z
        map_C5 = np.zeros((16, 4))
        map_C5[10, 0] = 1
        map_C5[11, 1] = 1
        map_C5[14, 2] = 1
        map_C5[15, 3] = 1
        
        add_block(35, map_A5, map_B5, map_C5)
        
        # Block 6: X = A12 - A22, Y = B21 + B22
        # X_00 = A_02 - A_22
        # X_01 = A_03 - A_23
        # X_10 = A_12 - A_32
        # X_11 = A_13 - A_33
        map_A6 = np.zeros((4, 16))
        map_A6[0, 2] = 1; map_A6[0, 10] = -1
        map_A6[1, 3] = 1; map_A6[1, 11] = -1
        map_A6[2, 6] = 1; map_A6[2, 14] = -1
        map_A6[3, 7] = 1; map_A6[3, 15] = -1
        
        map_B6 = np.zeros((4, 16))
        map_B6[0, 8] = 1; map_B6[0, 10] = 1
        map_B6[1, 9] = 1; map_B6[1, 11] = 1
        map_B6[2, 12] = 1; map_B6[2, 14] = 1
        map_B6[3, 13] = 1; map_B6[3, 15] = 1
        
        # C22 += Z
        map_C6 = np.zeros((16, 4))
        map_C6[10, 0] = 1
        map_C6[11, 1] = 1
        map_C6[14, 2] = 1
        map_C6[15, 3] = 1
        
        add_block(42, map_A6, map_B6, map_C6)
        
        return U_full, V_full, W_full
    
    def get_schoolbook_4x4x1():
        # Returns U, V, W for 4x4x1 (rank 16)
        # U shape: (16, 16)
        # V shape: (4, 16)  (B is 4x1)
        # W shape: (4, 16)  (C is 4x1)
        
        U = [[0]*16 for _ in range(16)]
        V = [[0]*16 for _ in range(4)]
        W = [[0]*16 for _ in range(4)]
        
        col = 0
        for i in range(4):
            for j in range(4):
                U[i*4+j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W

    # Assemble 4x4x9
    # We will use Strassen for the first 4 columns of the output (4x4x4 block).
    # Strassen for the next 4 columns (4x4x4 block).
    # Schoolbook for the last 1 column (4x4x1 block).
    
    rank_4x4x4 = 49
    rank_4x4x1 = 16
    
    total_rank = rank_4x4x4 + rank_4x4x4 + rank_4x4x1
    # 49 + 49 + 16 = 114
    
    U_total = [[0]*total_rank for _ in range(n*m)] # 16 x 114
    V_total = [[0]*total_rank for _ in range(m*p)] # 36 x 114
    W_total = [[0]*total_rank for _ in range(p*n)] # 36 x 114
    
    # Block 1: Columns 0-3 of output (B cols 0-3)
    U1, V1, W1 = get_strassen_4x4()
    # V1 is 16x49. We need to embed it into V_total which is 36x114.
    # V1 rows correspond to B[0..3, 0..3].
    # In V_total, rows 0..15 correspond to B[0..3, 0..3].
    # So we copy V1[i][j] to V_total[i][j].
    
    col_offset = 0
    for i in range(16):
        for j in range(49):
            U_total[i][col_offset + j] = U1[i][j]
            V_total[i][col_offset + j] = V1[i][j]
            # W1 is 16x49. Rows 0..15 correspond to C[0..3, 0..3].
            # In W_total, rows are indexed by k*n+i where k is col index of B, i is row index of A.
            # Wait, the trace convention is W[k*n+i, r].
            # Here k is the column index of the product C (which comes from B's column).
            # i is the row index of the product C (which comes from A's row).
            # In our 4x4x4 block, B has cols 0-3. So k in 0..3.
            # A has rows 0-3. So i in 0..3.
            # The W1 matrix rows are indexed by C index.
            # In the 4x4x4 decomposition, the output is 4x4.
            # The row index of W1 is `out_row`, which corresponds to `i` in the global trace convention?
            # No, let's look at the schoolbook code:
            # W[k * n + i][column] = 1
            # k is from B's column, i is from A's row.
            # In get_strassen_4x4, W_full[i] corresponds to C[i].
            # C[i] is the i-th row of the output.
            # So i in 0..3 is the row index of A.
            # But we also need the column index of B (k).
            # In the 4x4x4 block, the output C is 4x4.
            # C[r, c] = sum A[r, k] B[k, c].
            # The rank-1 term contributes to C[r, c] via W[c*4 + r]?
            # Let's check the schoolbook:
            # U[i*m+j][c] = 1 (A[i, j])
            # V[j*p+k][c] = 1 (B[j, k])
            # W[k*n+i][c] = 1 (C[i, k])
            # So W index is k*n + i. k is B's col, i is A's row.
            # In our Strassen 4x4x4, the output is 4x4.
            # The rows of W_full are 0..15.
            # We need to map these to the global W indices.
            # The global W index for this block is k*4 + i, where k in 0..3 (B cols 0-3), i in 0..3 (A rows 0-3).
            # But W_full[i] in get_strassen_4x4 is just the i-th row of the 4x4 output.
            # This implies that the Strassen decomposition I built assumes the output is a single vector?
            # No, get_strassen_4x4 returns W_full of shape 16x49.
            # The rows of W_full are indexed 0..15.
            # In the construction, map_C0 etc mapped Z_r to C[i].
            # C[i] is the i-th entry of the flattened 4x4 output.
            # If we flatten C in row-major order (C[0,0], C[0,1], C[0,2], C[0,3], C[1,0]...),
            # then index i in W_full corresponds to C[i//4, i%4].
            # So i//4 is the row of C (which is row of A), and i%4 is the col of C (which is col of B).
            # So global W index = (i%4)*4 + (i//4).
            
            w_idx = i
            b_col = w_idx % 4
            a_row = w_idx // 4
            global_w_idx = b_col * n + a_row
            W_total[global_w_idx][col_offset + j] = W1[i][j]
    
    # Block 2: Columns 4-7 of output (B cols 4-7)
    # B is 4x9. Cols 4-7 are the second 4x4 block.
    # V_total rows for this block:
    # V_total row index = j*p + k.
    # j in 0..3 (A col / B row). k in 4..7 (B col).
    # So row index = j*9 + k.
    # k = 4..7.
    # Row indices:
    # j=0, k=4 -> 4
    # j=0, k=5 -> 5
    # j=0, k=6 -> 6
    # j=0, k=7 -> 7
    # j=1, k=4 -> 13
    # ...
    
    U2, V2, W2 = get_strassen_4x4()
    col_offset = 49
    for i in range(16):
        for j in range(49):
            U_total[i][col_offset + j] = U2[i][j]
            
            # Map V2 row i to V_total row.
            # V2 row i corresponds to B_local[i//4, i%4].
            # B_local is 4x4.
            # b_row = i // 4, b_col_local = i % 4.
            # b_col_global = 4 + b_col_local.
            # V_total row = b_row * 9 + b_col_global.
            
            b_row = i // 4
            b_col_local = i % 4
            b_col_global = 4 + b_col_local
            v_total_idx = b_row * 9 + b_col_global
            V_total[v_total_idx][col_offset + j] = V2[i][j]
            
            # Map W2 row i to W_total row.
            # W2 row i corresponds to C_local[i//4, i%4].
            # c_row = i // 4 (A row).
            # c_col = i % 4 (B col local).
            # c_col_global = 4 + c_col.
            # global_w_idx = c_col_global * 4 + c_row.
            
            c_row = i // 4
            c_col = i % 4
            c_col_global = 4 + c_col
            global_w_idx = c_col_global * 4 + c_row
            W_total[global_w_idx][col_offset + j] = W2[i][j]
    
    # Block 3: Column 8 of output (B col 8)
    # 4x4x1 block.
    U3, V3, W3 = get_schoolbook_4x4x1()
    col_offset = 98
    for i in range(16):
        for j in range(16):
            U_total[i][col_offset + j] = U3[i][j]
            
            # V3 row i corresponds to B_local[i, 0] (since it's 4x1).
            # b_row = i, b_col_local = 0.
            # b_col_global = 8.
            # v_total_idx = i * 9 + 8.
            
            b_row = i
            b_col_global = 8
            v_total_idx = b_row * 9 + b_col_global
            V_total[v_total_idx][col_offset + j] = V3[i][j]
            
            # W3 row i corresponds to C_local[i, 0].
            # c_row = i, c_col = 0.
            # c_col_global = 8.
            # global_w_idx = 8 * 4 + i.
            
            c_row = i
            c_col_global = 8
            global_w_idx = c_col_global * 4 + c_row
            W_total[global_w_idx][col_offset + j] = W3[i][j]
    
    return {"U": U_total, "V": V_total, "W": W_total}
# EVOLVE-BLOCK-END
