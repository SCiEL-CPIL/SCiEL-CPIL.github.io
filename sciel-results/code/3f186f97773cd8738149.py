"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We use Strassen's algorithm recursively to achieve rank 56.
    # Strassen decomposition for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Output:
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6
    
    # We have A (2x4), B (4x8), C (2x8).
    # Split A into A_left (2x2) and A_right (2x2).
    # Split B into B_tl (2x4), B_tr (2x4), B_bl (2x4), B_br (2x4).
    # C_left (2x4) = A_left * B_tl + A_right * B_bl
    # C_right (2x4) = A_left * B_tr + A_right * B_br
    
    # For each 2x2x4 product, split the 4 columns into two 2x2x2 products.
    # So we have 4 sub-problems of 2x2x2, each needing rank 7. Total rank = 4 * 7 = 28? 
    # Wait, Strassen for 2x2x2 has rank 7.
    # Let's verify the structure carefully.
    
    # Actually, let's build the Strassen tensors for each 2x2x2 block explicitly.
    # We have 4 such blocks:
    # 1. A_left (cols 0,1) * B_tl (rows 0,1; cols 0,1) -> contributes to C(0,0), C(0,1), C(1,0), C(1,1)
    # 2. A_left (cols 0,1) * B_tr (rows 0,1; cols 2,3) -> contributes to C(0,2), C(0,3), C(1,2), C(1,3)
    # 3. A_right (cols 2,3) * B_bl (rows 2,3; cols 0,1) -> contributes to C(0,0), C(0,1), C(1,0), C(1,1)
    # 4. A_right (cols 2,3) * B_br (rows 2,3; cols 2,3) -> contributes to C(0,2), C(0,3), C(1,2), C(1,3)
    
    # Wait, the output C is 2x8.
    # C_left is cols 0-3 of C. C_right is cols 4-7 of C.
    # So:
    # Block 1: A_left (2x2) * B_tl (2x4 where B_tl rows 0,1, cols 0,1). This gives 2x2, placed at C[0:2, 0:2].
    # Block 2: A_left (2x2) * B_tr (2x4 where B_tr rows 0,1, cols 2,3). This gives 2x2, placed at C[0:2, 2:4].
    # Block 3: A_right (2x2) * B_bl (2x4 where B_bl rows 2,3, cols 0,1). This gives 2x2, placed at C[0:2, 0:2].
    # Block 4: A_right (2x2) * B_br (2x4 where B_br rows 2,3, cols 2,3). This gives 2x2, placed at C[0:2, 2:4].
    
    # Let's define the Strassen 7-term decomposition for 2x2x2.
    # Input A (2x2), B (2x2).
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    
    # U matrix rows: 4 (a00, a01, a10, a11)
    # V matrix rows: 4 (b00, b01, b10, b11)
    # W matrix rows: 4 (c00, c01, c10, c11)
    
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22)  -> (a00+a11)(b00+b11)
    # M2 = (A21+A22)B11        -> (a10+a11)b00
    # M3 = A11(B12-B22)        -> a00(b01-b11)
    # M4 = A22(B21-B11)        -> a11(b10-b00)
    # M5 = (A11+A12)B22        -> (a00+a01)b11
    # M6 = (A21-A11)(B11+B12)  -> (a10-a00)(b00+b01)
    # M7 = (A12-A22)(B21+B22)  -> (a01-a11)(b10+b11)
    
    # Output reconstruction:
    # c00 = M1 - M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 - M6
    
    # We need to map these to the global indices.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
    # We'll build 4 sets of 7 terms.
    def add_strassen(A_row_offsets, B_row_offsets, C_col_offsets):
        # A is 2x2, indexed by A_row_offsets (two row indices in global A)
        # B is 2x4, but we split into 2x2 and 2x2. 
        # Actually, let's just say we're computing 2x2 * 2x2.
        # A_row_offsets: [i1, i2] for the 2 rows of the A block.
        # B_row_offsets: [j1, j2] for the 2 rows of the B block.
        # B_col_offsets: [k1, k2] for the 2 cols of the B block.
        # C_row_offsets: [i1, i2] for the 2 rows of the C block.
        # C_col_offsets: [k1, k2] for the 2 cols of the C block.
        
        pass

    # Let's do it more directly. We'll define the 4 sub-blocks.
    # Each sub-block is 2x2x2.
    # Sub-block 1: A rows 0,1, cols 0,1; B rows 0,1, cols 0,1; C rows 0,1, cols 0,1.
    # Sub-block 2: A rows 0,1, cols 0,1; B rows 0,1, cols 2,3; C rows 0,1, cols 2,3.
    # Sub-block 3: A rows 0,1, cols 2,3; B rows 2,3, cols 0,1; C rows 0,1, cols 0,1.
    # Sub-block 4: A rows 0,1, cols 2,3; B rows 2,3, cols 2,3; C rows 0,1, cols 2,3.
    
    # Wait, A is 2x4. A_row_indices are 0,1. A_col_indices are 0,1,2,3.
    # B is 4x8. B_row_indices are 0,1,2,3. B_col_indices are 0..7.
    # C is 2x8. C_row_indices are 0,1. C_col_indices are 0..7.
    
    # Sub-block 1: A[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: (0,0), (0,1), (1,0), (1,1) -> global U rows: 0, 1, 4, 5
    # B indices: (0,0), (0,1), (1,0), (1,1) -> global V rows: 0, 1, 4, 5
    # C indices: (0,0), (0,1), (1,0), (1,1) -> global W rows: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3
    # W[k*n+i, r]. For n=2: W[0*2+0]=W[0], W[0*2+1]=W[1], W[1*2+0]=W[2], W[1*2+1]=W[3]. Correct.
    
    def add_strassen_block(U_rows, V_rows, W_rows, rank_offset):
        # U_rows: [u00, u01, u10, u11] corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
        # V_rows: [v00, v01, v10, v11] corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
        # W_rows: [w00, w01, w10, w11] corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
        
        u00, u01, u10, u11 = U_rows
        v00, v01, v10, v11 = V_rows
        w00, w01, w10, w11 = W_rows
        
        # M1 = (a00+a11)(b00+b11)
        c = rank_offset
        U[u00][c] += 1
        U[u11][c] += 1
        V[v00][c] += 1
        V[v11][c] += 1
        W[w00][c] += 1  # c00 += M1
        W[w11][c] += 1  # c11 += M1
        c += 1
        
        # M2 = (a10+a11)b00
        U[u10][c] += 1
        U[u11][c] += 1
        V[v00][c] += 1
        W[w10][c] += 1  # c10 += M2
        W[w11][c] += -1 # c11 += -M2
        c += 1
        
        # M3 = a00(b01-b11)
        U[u00][c] += 1
        V[v01][c] += 1
        V[v11][c] += -1
        W[w01][c] += 1  # c01 += M3
        W[w11][c] += 1  # c11 += M3
        c += 1
        
        # M4 = a11(b10-b00)
        U[u11][c] += 1
        V[v10][c] += 1
        V[v00][c] += -1
        W[w00][c] += -1 # c00 += -M4
        W[w10][c] += 1  # c10 += M4
        c += 1
        
        # M5 = (a00+a01)b11
        U[u00][c] += 1
        U[u01][c] += 1
        V[v11][c] += 1
        W[w00][c] += -1 # c00 += -M5
        W[w01][c] += 1  # c01 += M5
        c += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u10][c] += 1
        U[u00][c] += -1
        V[v00][c] += 1
        V[v01][c] += 1
        W[w11][c] += -1 # c11 += -M6
        c += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u01][c] += 1
        U[u11][c] += -1
        V[v10][c] += 1
        V[v11][c] += 1
        W[w00][c] += 1  # c00 += M7
        c += 1
        
        return rank_offset + 7

    current_rank = 0
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U_rows1 = [0, 1, 4, 5]  # A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    V_rows1 = [0, 1, 4, 5]  # B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    W_rows1 = [0, 1, 2, 3]  # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    current_rank += add_strassen_block(U_rows1, V_rows1, W_rows1, current_rank)
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Wait, B is 4x8. B[0:2, 2:4] means rows 0,1, cols 2,3.
    V_rows2 = [2, 3, 6, 7]  # B[0,2]=0*4+2=2, B[0,3]=3, B[1,2]=1*4+2=6, B[1,3]=7
    W_rows2 = [4, 5, 6, 7]  # C[0,2]=0*2+2=4? No, W[k*n+i]. C[i,k] -> W[k*n+i].
    # C[0,2] -> k=2, i=0 -> W[2*2+0]=W[4]. Correct.
    # C[0,3] -> k=3, i=0 -> W[3*2+0]=W[6].
    # C[1,2] -> k=2, i=1 -> W[2*2+1]=W[5].
    # C[1,3] -> k=3, i=1 -> W[3*2+1]=W[7].
    W_rows2 = [4, 6, 5, 7]  # w00, w01, w10, w11 correspond to C[0,2], C[0,3], C[1,2], C[1,3]
    current_rank += add_strassen_block(U_rows1, V_rows2, W_rows2, current_rank)
    
    # Block 3: A[:, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    U_rows3 = [2, 3, 6, 7]  # A[0,2]=0*4+2=2, A[0,3]=3, A[1,2]=1*4+2=6, A[1,3]=7
    V_rows3 = [0, 1, 4, 5]  # B[2,0]=2*4+0=8? No, V row is j*p+k. p=8.
    # V[j*p+k]. B row j, col k.
    # B[2,0] -> j=2, k=0 -> 2*8+0 = 16.
    # B[2,1] -> j=2, k=1 -> 17.
    # B[3,0] -> j=3, k=0 -> 24.
    # B[3,1] -> j=3, k=1 -> 25.
    V_rows3 = [16, 17, 24, 25]
    W_rows3 = [0, 1, 2, 3]
    current_rank += add_strassen_block(U_rows3, V_rows3, W_rows3, current_rank)
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # B[2,2] -> j=2, k=2 -> 18
    # B[2,3] -> j=2, k=3 -> 19
    # B[3,2] -> j=3, k=2 -> 26
    # B[3,3] -> j=3, k=3 -> 27
    V_rows4 = [18, 19, 26, 27]
    W_rows4 = [4, 6, 5, 7]
    current_rank += add_strassen_block(U_rows3, V_rows4, W_rows4, current_rank)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

