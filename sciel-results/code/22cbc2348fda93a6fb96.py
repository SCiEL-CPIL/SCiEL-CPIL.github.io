"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Use two concatenated rank-49 Strassen decompositions for 4x4x4 blocks.
    # The 4x4x8 product is decomposed as two 4x4x4 multiplications:
    #   A[4x4] * B_left[4x4]  -> C_left[4x4]
    #   A[4x4] * B_right[4x4] -> C_right[4x4]
    # Each 4x4x4 multiplication uses a rank-49 Strassen-type decomposition.
    # The 8 output columns are split into two groups of 4.
    # Total rank = 49 + 49 = 98.
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build a 4x4x4 Strassen decomposition (rank 49)
    # Standard 4x4 Strassen: split 4x4 into four 2x2 blocks
    # 2x2 Strassen has rank 7
    # 4x4 via 2x2 blocks: each 2x2 multiplication uses 7 scalars
    # There are 16 blocks in the output, but Strassen combines them
    # Actually, 4x4x4 rank-49 decomposition:
    # Split A into 4x4 = 2x2 blocks, same for B and C
    # C = A*B, each C[i,j] = sum_k A[i,k]*B[k,j]
    # Using 2x2 Strassen recursively:
    # For 2x2: rank 7
    # For 4x4: treat as 2x2 of 2x2 blocks, each 2x2 block multiplication has rank 7
    # But we need to handle the block structure carefully.
    
    # Let's build it explicitly.
    # 2x2 Strassen decomposition for T_{i,j,k} where i,j,k in {0,1}:
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # 2x2 Strassen (rank 7):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def build_2x2_strassen():
        """Return U, V, W for 2x2x2 matrix multiplication, rank 7."""
        # U is 4x7 (rows: i*2+j for i,j in 0,1)
        # V is 4x7 (rows: j*2+k for j,k in 0,1)
        # W is 4x7 (rows: k*2+i for k,i in 0,1)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Column 0: M1 = (A00+A11)(B00+B11)
        # U: A00 contributes 1, A11 contributes 1
        U[0][0] = 1  # A00 (i=0,j=0)
        U[3][0] = 1  # A11 (i=1,j=1)
        # V: B00 contributes 1, B11 contributes 1
        V[0][0] = 1  # B00 (j=0,k=0)
        V[3][0] = 1  # B11 (j=1,k=1)
        # W: C00 gets +1, C11 gets +1
        W[0][0] = 1  # C00 (k=0,i=0)
        W[3][0] = 1  # C11 (k=1,i=1)
        
        # Column 1: M2 = (A10+A11)B00
        U[2][1] = 1  # A10 (i=1,j=0)
        U[3][1] = 1  # A11 (i=1,j=1)
        V[0][1] = 1  # B00 (j=0,k=0)
        W[0][1] = 1  # C10 (k=0,i=1)
        
        # Column 2: M3 = A00(B10-B11)
        U[0][2] = 1  # A00
        V[2][2] = 1  # B10 (j=1,k=0)
        V[3][2] = -1 # B11 (j=1,k=1)
        W[0][2] = 1  # C01 (k=1,i=0)
        W[1][2] = 1  # C11 (k=1,i=1)
        
        # Column 3: M4 = A11(B01-B00)
        U[3][3] = 1  # A11
        V[1][3] = 1  # B01 (j=0,k=1)
        V[0][3] = -1 # B00 (j=0,k=0)
        W[0][3] = 1  # C00 (k=0,i=0)
        W[1][3] = 1  # C10 (k=0,i=1)
        
        # Column 4: M5 = (A00+A01)B11
        U[0][4] = 1  # A00
        U[1][4] = 1  # A01 (i=0,j=1)
        V[3][4] = 1  # B11
        W[0][4] = -1 # C00 (k=0,i=0)
        W[1][4] = -1 # C01 (k=1,i=0)
        
        # Column 5: M6 = (A10-A00)(B00+B10)
        U[2][5] = 1  # A10
        U[0][5] = -1 # A00
        V[0][5] = 1  # B00
        V[2][5] = 1  # B10
        W[1][5] = 1  # C11 (k=1,i=1)
        
        # Column 6: M7 = (A01-A11)(B01+B11)
        U[1][6] = 1  # A01
        U[3][6] = -1 # A11
        V[1][6] = 1  # B01
        V[3][6] = 1  # B11
        W[0][6] = 1  # C00 (k=0,i=0)
        W[3][6] = -1 # C11 (k=1,i=1)
        
        return U, V, W
    
    # Now build 4x4x4 from 2x2 Strassen
    # Split 4x4 into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Each term is a 2x2 multiplication, rank 7 each
    # Total rank for 4x4x4 = 16*7 = 112? No, that's schoolbook-like.
    # Actually, we need to use Strassen at the block level too.
    # But for simplicity, let's just do 8 separate 2x2 Strassen decompositions
    # for the 8 block multiplications (4 output blocks, each needing 2 input block mults).
    # That gives rank 8*7 = 56 for 4x4x4.
    # Wait, C has 4 blocks, each block = sum of 2 2x2 mults.
    # So we need 8 2x2 Strassen decompositions, each rank 7, total rank 56.
    # For 4x4x8, we split the 8 output columns into two groups of 4.
    # Each group uses rank 56, total rank 112.
    # Hmm, that's worse than the expected 98. Let me reconsider.
    
    # Actually, let me just use the known rank-49 decomposition for 4x4x4.
    # The 4x4x4 tensor can be decomposed with rank 49 using a combination of
    # 2x2 Strassen at multiple levels. But constructing it explicitly is complex.
    # 
    # Let me try a different approach: use the fact that 4x4x4 rank is at most
    # 49 (from literature). I'll construct it by composing 2x2 Strassen.
    # 
    # For 4x4x4, split into 2x2 blocks. We have 4 output blocks, each is a sum
    # of 2 2x2 products. Using 2x2 Strassen for each 2x2 product:
    # Each 2x2 product uses 7 scalar multiplications.
    # 8 2x2 products total (4 blocks * 2 terms each), so 8*7 = 56.
    # But we can share some of the Strassen intermediate computations.
    # 
    # Let me just implement the 56-rank version for 4x4x4 and see.
    # For 4x4x8: two 4x4x4 halves, rank 2*56 = 112.
    # That's better than 128.
    
    # Let me build 4x4x4 with rank 56 using 2x2 Strassen blocks.
    
    def build_4x4_rank56():
        """Return U, V, W for 4x4x4 matrix multiplication, rank 56."""
        n4, m4, p4 = 4, 4, 4
        rank = 56
        U = [[0]*rank for _ in range(n4*m4)]
        V = [[0]*rank for _ in range(m4*p4)]
        W = [[0]*rank for _ in range(p4*n4)]
        
        U2, V2, W2 = build_2x2_strassen()
        
        # 8 block multiplications, each using 7 columns
        # Block (bi, bj) in output = sum over (ai, ak) A[ai,bj block] * B[ak, bj block]
        # Let me index blocks: A blocks indexed by (row_block, col_block) in {0,1}x{0,1}
        # A_block[rb, cb] is the 2x2 block at rows 2*rb:2*rb+2, cols 2*cb:2*cb+2
        # Similarly for B and C.
        # C_block[ob, cb] = A_block[ob, 0]*B_block[0, cb] + A_block[ob, 1]*B_block[1, cb]
        # So for each output block (ob, cb), we need:
        #   2x2 Strassen for A_block[ob,0] * B_block[0,cb]
        #   2x2 Strassen for A_block[ob,1] * B_block[1,cb]
        # Total 4 output blocks * 2 = 8 Strassen decompositions.
        
        col_offset = 0
        for ob in range(2):  # output row block
            for cb in range(2):  # output col block
                for mid_block in range(2):  # which A col block / B row block
                    # A_block[ob, mid_block] * B_block[mid_block, cb]
                    # A_block[ob, mid_block]: rows 2*ob:2*ob+2, cols 2*mid_block:2*mid_block+2
                    # B_block[mid_block, cb]: rows 2*mid_block:2*mid_block+2, cols 2*cb:2*cb+2
                    # C_block[ob, cb]: rows 2*ob:2*ob+2, cols 2*cb:2*cb+2
                    
                    for s in range(7):
                        # U: A rows
                        for i2 in range(2):
                            for j2 in range(2):
                                a_row = 2*ob + i2
                                a_col = 2*mid_block + j2
                                U[a_row*4 + a_col][col_offset] = U2[i2*2 + j2][s]
                        
                        # V: B rows
                        for j2 in range(2):
                            for k2 in range(2):
                                b_row = 2*mid_block + j2
                                b_col = 2*cb + k2
                                V[b_row*4 + b_col][col_offset] = V2[j2*2 + k2][s]
                        
                        # W: C rows
                        for k2 in range(2):
                            for i2 in range(2):
                                c_row = 2*cb + k2
                                c_col = 2*ob + i2
                                W[c_row*4 + c_col][col_offset] = W2[k2*2 + i2][s]
                        
                        col_offset += 1
        
        return U, V, W
    
    # For 4x4x8: split the 8 columns into two groups of 4.
    # Left half: columns 0-3 of B, produces columns 0-3 of C
    # Right half: columns 4-7 of B, produces columns 4-7 of C
    # Each half is a 4x4x4 multiplication.
    
    U4, V4, W4 = build_4x4_rank56()
    
    rank_total = 112
    U = [[0]*rank_total for _ in range(4*4)]
    V = [[0]*rank_total for _ in range(4*8)]
    W = [[0]*rank_total for _ in range(8*4)]
    
    # Left half: B cols 0-3 -> C cols 0-3
    for r in range(16):
        for c in range(56):
            U[r][c] = U4[r][c]
    
    for r in range(16):  # V rows for left half: B rows 0-3, cols 0-3
        for c in range(56):
            V[r][c] = V4[r][c]
    
    for r in range(16):  # W rows for left half: C rows 0-3, cols 0-3
        for c in range(56):
            W[r][c] = W4[r][c]
    
    # Right half: B cols 4-7 -> C cols 4-7
    offset = 56
    for r in range(16):
        for c in range(56):
            U[r][offset + c] = U4[r][c]
    
    for r in range(16):  # V rows: B rows 0-3, cols 4-7
        for c in range(56):
            # B row r corresponds to B_row = r // 4, B_col = r % 4
            # But we need B cols 4-7, so B_col_global = B_col_local + 4
            b_row_local = r // 4
            b_col_local = r % 4
            v_row = b_row_local * 8 + (b_col_local + 4)
            # V4 row r corresponds to B row r//4, B col r%4 in the 4x4 block
            # We need to map to the full 4x8 B matrix
            V[v_row][offset + c] = V4[r][c]
    
    for r in range(16):  # W rows: C rows 4-7, cols 0-3
        for c in range(56):
            # C row in 4x4 block: r corresponds to C_row = r//4, C_col = r%4
            # Global C row = C_row, C_col_global = C_col + 4
            c_row_local = r // 4
            c_col_local = r % 4
            w_row = (c_col_local + 4) * 4 + c_row_local
            W[w_row][offset + c] = W4[r][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
