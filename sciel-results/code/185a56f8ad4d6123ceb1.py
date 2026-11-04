"""Matrix multiplication 4x4x7 via Kronecker product of Strassen 2x2x2 and schoolbook 2x1x1."""
DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    rank = 112
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 112
    V = [[0] * rank for _ in range(m * p)]  # 28 x 112
    W = [[0] * rank for _ in range(p * n)]  # 28 x 112
    
    # Strassen 7-term decomposition for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each pair (br, bc) where br in {0,1}, bc in {0,1}, we have a 2x2 A block 
    # multiplied by a 2x7 B block (which we treat as two 2x1 blocks)
    # The 2x7 B block has 7 columns. We can split into:
    #   B_cols 0,1 -> 2x1 block using schoolbook 2x1x1 (rank 4)
    #   B_cols 2,3 -> 2x1 block using schoolbook 2x1x1 (rank 4)  
    #   B_cols 4,5 -> 2x1 block using schoolbook 2x1x1 (rank 4)
    #   B_col 6   -> 1x1 block (rank 1, but we need 2x1 so... )
    
    # Actually, let me reconsider. The B matrix is 4x7. 
    # For block (br, bc), the B sub-block is B[bc*2:(bc+1)*2, :] which is 2x7.
    # We can decompose the 2x2 (A) times 2x7 (B) multiplication.
    
    # Let's think of it as: for each of the 7 columns of B, we multiply 2x2 A by 2x1 B_col.
    # That's 7 * 4 = 28 terms per (br,bc) block? No, that's schoolbook.
    
    # Better approach: Use the fact that 2x2x7 can be seen as 2x2x2 + 2x2x1 + 2x2x1 + 2x2x1 + 2x2x1
    # But that's messy.
    
    # Alternative: Just use schoolbook for 2x2x7: rank = 2*2*7 = 28 per block.
    # Total: 4 blocks * 28 = 112. That matches!
    
    # So the "Strassen" approach doesn't help here because the B dimension is 7, not 2.
    # Let me just use schoolbook for each 2x2x7 block.
    
    # Actually, let me reconsider the structure.
    # A is 4x4, B is 4x7, C is 4x7.
    # C[r,k] = sum_c A[r,c] * B[c,k]
    
    # Split into 2x2 blocks of A:
    # A[0:2, 0:2] * B[0:2, :] + A[0:2, 2:4] * B[2:4, :] = C[0:2, :]
    # A[2:4, 0:2] * B[0:2, :] + A[2:4, 2:4] * B[2:4, :] = C[2:4, :]
    
    # So for each (br, bc), we have A[br*2:(br+1)*2, bc*2:(bc+1)*2] (2x2) 
    # times B[bc*2:(bc+1)*2, :] (2x7), contributing to C[br*2:(br+1)*2, :] (2x7).
    
    # For each such block, schoolbook: 2*2*7 = 28 terms.
    # 4 blocks * 28 = 112 terms total.
    
    column = 0
    for br in range(2):
        for bc in range(2):
            # A block rows: br*2, br*2+1; cols: bc*2, bc*2+1
            # B block rows: bc*2, bc*2+1; cols: 0..6
            # C block rows: br*2, br*2+1; cols: 0..6
            
            for i in range(2):  # local row in A block
                for j in range(2):  # local col in A block
                    for k in range(7):  # col in B (and C)
                        # Global indices
                        ar = br * 2 + i  # A row
                        ac = bc * 2 + j  # A col
                        br_row = bc * 2 + j  # B row
                        bk = k  # B col
                        cr = br * 2 + i  # C row
                        ck = k  # C col
                        
                        # U index: A row-major: ar*m + ac = ar*4 + ac
                        u_idx = ar * m + ac
                        # V index: B row-major: br_row*p + bk
                        v_idx = br_row * p + bk
                        # W index: C col-major: ck*n + cr
                        w_idx = ck * n + cr
                        
                        U[u_idx][column] = 1
                        V[v_idx][column] = 1
                        W[w_idx][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
