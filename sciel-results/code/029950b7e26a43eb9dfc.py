"""Matrix multiplication 2x3x8 with rank-44 decomposition using Strassen + direct sub-blocks."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    # We split the 3x8 matrix into two 3x4 blocks: B1 (cols 0-3), B2 (cols 4-7)
    # For each 3x4 block, we further split into:
    #   - a 2x2 sub-block (rows 0-1, cols 0-1 of the 3x4 block) -> use Strassen (rank 7)
    #   - remaining: row 2 with cols 0-1 (2x1x2), rows 0-1 with cols 2-3 (2x2x2... wait)
    #
    # Let me reconsider the block strategy.
    # A is 2x3, B is 3x8.
    # Split B into two 3x4 blocks: B = [B1 | B2] where B1, B2 are 3x4.
    # Then C = A*B = [A*B1 | A*B2], where each A*Bj is 2x4.
    #
    # For each 3x4 block Bj, split it as:
    #   Bj = [Bj_top | Bj_right] where Bj_top is 3x2 (cols 0,1) and Bj_right is 3x2 (cols 2,3)
    # Actually, let's split the 3x4 into:
    #   - 2x2 top-left: rows 0-1, cols 0-1 -> Strassen (rank 7 per such block, but we need to handle carefully)
    #   - The rest
    #
    # Alternative cleaner approach:
    # Split the 3 columns of A (and rows of B) into:
    #   Group 1: column 0 (1x1)
    #   Group 2: columns 1-2 (2x2)
    # And split the 8 columns of B into two groups of 4:
    #   Group A: cols 0-3
    #   Group B: cols 4-7
    #
    # So we have 2x2 sub-problems:
    #   (1 col A) x (4 cols B) -> rank 4 (direct)
    #   (2 cols A) x (4 cols B) -> split further
    #
    # For (2 cols A) x (4 cols B): split the 4 cols into 2+2
    #   (2x2) x (2x2) -> Strassen rank 7
    #   (2x2) x (2x2) -> Strassen rank 7
    # So (2 cols A) x (4 cols B) = 7 + 7 = 14
    # (1 col A) x (4 cols B) = 4 + 4 = 8 (two 1x1 x 2x2 blocks, each rank 2? No, 1x1x2 = rank 2)
    # Wait: (1 col) x (4 cols) means A is 2x1, B is 1x4. That's rank 2*1*4=8.
    # Actually for (2x1) x (1x4): rank = 2*4 = 8.
    #
    # Total: for each of the 4-col groups:
    #   (2x1) x (1x4) = rank 8
    #   (2x2) x (2x4) = 7+7 = 14
    # Per 4-col group: 8 + 14 = 22
    # Two 4-col groups: 22 + 22 = 44
    #
    # Let me verify: A is 2x3, split columns into [col0] and [col1,col2]
    # B is 3x8, split rows correspondingly: [row0] and [row1,row2], and split cols into [0-3] and [4-7]
    #
    # Block (A_col0, B_row0, B_cols0-3): (2x1) x (1x4) -> rank 8
    # Block (A_col0, B_row0, B_cols4-7): (2x1) x (1x4) -> rank 8
    # Block (A_cols1-2, B_rows1-2, B_cols0-3): (2x2) x (2x4) -> split into (2x2)x(2x2) + (2x2)x(2x2) = 7+7=14
    # Block (A_cols1-2, B_rows1-2, B_cols4-7): (2x2) x (2x4) -> 7+7=14
    # Total: 8+8+14+14 = 44
    #
    # Now implement this.
    #
    # Indexing:
    # U has rows indexed by (i, j) where i in [0,2), j in [0,3) -> U[i*3+j]
    # V has rows indexed by (j, k) where j in [0,3), k in [0,8) -> V[j*8+k]
    # W has rows indexed by (k, i) where k in [0,8), i in [0,2) -> W[k*2+i]
    #
    # For each rank-1 term, we set one column of U, V, W.
    #
    # Direct multiplication for (2x1) x (1x4):
    #   A is 2x1 (rows i=0,1; col j), B is 1x4 (row j; cols k0,k1,k2,k3)
    #   C is 2x4: C[i,k] = A[i,j] * B[j,k]
    #   Rank-1 terms: for each (i,k): U[i,j]=1, V[j,k]=1, W[k,i]=1 -> 2*4=8 terms
    #
    # Strassen for (2x2) x (2x2):
    #   A is 2x2: A[i,j] for i in [0,2), j in [0,2) (local indices)
    #   B is 2x2: B[j,k] for j in [0,2), k in [0,2) (local indices)
    #   C is 2x2: C[i,k] for i in [0,2), k in [0,2)
    #
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10+B11)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    #
    #   C00 = M1 - M2 + M4 + M5
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M3 + M6 + M7
    #
    # Each Mr is a rank-1 term. For each Mr, the U, V, W entries:
    #
    # M1: U gets contributions from A00 and A11; V gets from B00 and B11
    #   U[A00]=1, U[A11]=1; V[B00]=1, V[B11]=1
    #   W: C00 gets +1, C11 gets +1 -> W[C00]=1, W[C11]=1
    #
    # M2: U from A10, A11; V from B00
    #   U[A10]=1, U[A11]=1; V[B00]=1
    #   W: C10 gets +1, C00 gets -1 -> W[C10]=1, W[C00]=-1
    #
    # M3: U from A00; V from B01, B11
    #   U[A00]=1; V[B01]=1, V[B11]=-1
    #   W: C01 gets +1, C11 gets -1 -> W[C01]=1, W[C11]=-1
    #
    # M4: U from A11; V from B10, B11
    #   U[A11]=1; V[B10]=1, V[B11]=1
    #   W: C10 gets +1, C00 gets +1 -> W[C10]=1, W[C00]=1
    #
    # M5: U from A00, A01; V from B11
    #   U[A00]=1, U[A01]=1; V[B11]=1
    #   W: C00 gets +1, C01 gets +1 -> W[C00]=1, W[C01]=1
    #
    # M6: U from A10, A00; V from B00, B01
    #   U[A10]=1, U[A00]=-1; V[B00]=1, V[B01]=1
    #   W: C11 gets +1 -> W[C11]=1
    #
    # M7: U from A01, A11; V from B10, B11
    #   U[A01]=1, U[A11]=-1; V[B10]=1, V[B11]=1
    #   W: C11 gets +1 -> W[C11]=1
    #
    # Let me re-derive more carefully.
    # C = A*B
    # M1 = (A00+A11)*(B00+B11)
    #   In the tensor: M1 contributes to C via:
    #   C00 += M1 * 1 (coeff from expansion: C00 = M1 - M2 + M4 + M5, so M1 has coeff +1 in C00)
    #   C11 += M1 * 1 (C11 = M1 - M3 + M6 + M7, so M1 has coeff +1 in C11)
    #
    #   The rank-1 term for M1: U has nonzero at positions corresponding to (i=0,j=0) and (i=1,j=1) of the 2x2 A block, both with value 1.
    #   V has nonzero at (j=0,k=0) and (j=1,k=1) of the 2x2 B block, both with value 1.
    #   W has: for C00: coeff +1, for C11: coeff +1.
    #
    #   So W[C00_local] += 1, W[C11_local] += 1
    #   Where C00_local means k=0,i=0 in the 2x2 C block, C11_local means k=1,i=1.
    #
    # M2 = (A10+A11)*B00
    #   U: (i=1,j=0) val 1, (i=1,j=1) val 1
    #   V: (j=0,k=0) val 1
    #   W: C00 gets -1, C10 gets +1
    #   W[C00_local] += -1, W[C10_local] += 1
    #   C10_local: k=0, i=1
    #
    # M3 = A00*(B01-B11)
    #   U: (i=0,j=0) val 1
    #   V: (j=0,k=1) val 1, (j=1,k=1) val -1
    #   W: C01 gets +1, C11 gets -1
    #   C01_local: k=1, i=0
    #   W[C01_local] += 1, W[C11_local] += -1
    #
    # M4 = A11*(B10+B11)
    #   U: (i=1,j=1) val 1
    #   V: (j=1,k=0) val 1, (j=1,k=1) val 1
    #   W: C00 gets +1, C10 gets +1
    #   W[C00_local] += 1, W[C10_local] += 1
    #
    # M5 = (A00+A01)*B11
    #   U: (i=0,j=0) val 1, (i=0,j=1) val 1
    #   V: (j=1,k=1) val 1
    #   W: C00 gets +1, C01 gets +1
    #   W[C00_local] += 1, W[C01_local] += 1
    #
    # M6 = (A10-A00)*(B00+B01)
    #   U: (i=1,j=0) val 1, (i=0,j=0) val -1
    #   V: (j=0,k=0) val 1, (j=0,k=1) val 1
    #   W: C11 gets +1
    #   W[C11_local] += 1
    #
    # M7 = (A01-A11)*(B10+B11)
    #   U: (i=0,j=1) val 1, (i=1,j=1) val -1
    #   V: (j=1,k=0) val 1, (j=1,k=1) val 1
    #   W: C11 gets +1
    #   W[C11_local] += 1
    #
    # Now let's code this up.
    
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    def add_direct_2x1_x_1x4(i_offset, j_col, k_start):
        """Add 8 rank-1 terms for (2x1) x (1x4) block.
        A has rows i_offset, i_offset+1 (but only 1 col j_col), B has 1 row j_col, 4 cols k_start..k_start+3.
        C has 2 rows, 4 cols starting at k_start.
        """
        global col
        for i in range(2):
            for k in range(4):
                U[(i_offset + i) * m + j_col][col] = 1.0
                V[j_col * p + (k_start + k)][col] = 1.0
                W[(k_start + k) * n + (i_offset + i)][col] = 1.0
                col += 1
    
    def add_strassen_2x2_x_2x2(i_offset, j_offset, k_start):
        """Add 7 rank-1 terms for Strassen (2x2) x (2x2).
        A local indices: (i,j) -> global (i_offset+i, j_offset+j)
        B local indices: (j,k) -> global (j_offset+j, k_start+k)
        C local indices: (i,k) -> global (k_start+k, i_offset+i)
        """
        global col
        # M1 = (A00+A11)*(B00+B11)
        U[(i_offset+0)*m + (j_offset+0)][col] = 1.0
        U[(i_offset+1)*m + (j_offset+1)][col] = 1.0
        V[(j_offset+0)*p + (k_start+0)][col] = 1.0
        V[(j_offset+1)*p + (k_start+1)][col] = 1.0
        W[(k_start+0)*n + (i_offset+0)][col] = 1.0
        W[(k_start+1)*n + (i_offset+1)][col] = 1.0
        col += 1
        
        # M2 = (A10+A11)*B00
        U[(i_offset+1)*m + (j_offset+0)][col] = 1.0
        U[(i_offset+1)*m + (j_offset+1)][col] = 1.0
        V[(j_offset+0)*p + (k_start+0)][col] = 1.0
        W[(k_start+0)*n + (i_offset+0)][col] = -1.0
        W[(k_start+0)*n + (i_offset+1)][col] = 1.0
        col += 1
        
        # M3 = A00*(B01-B11)
        U[(i_offset+0)*m + (j_offset+0)][col] = 1.0
        V[(j_offset+0)*p + (k_start+1)][col] = 1.0
        V[(j_offset+1)*p + (k_start+1)][col] = -1.0
        W[(k_start+1)*n + (i_offset+0)][col] = 1.0
        W[(k_start+1)*n + (i_offset+1)][col] = -1.0
        col += 1
        
        # M4 = A11*(B10+B11)
        U[(i_offset+1)*m + (j_offset+1)][col] = 1.0
        V[(j_offset+1)*p + (k_start+0)][col] = 1.0
        V[(j_offset+1)*p + (k_start+1)][col] = 1.0
        W[(k_start+0)*n + (i_offset+0)][col] = 1.0
        W[(k_start+0)*n + (i_offset+1)][col] = 1.0
        col += 1
        
        # M5 = (A00+A01)*B11
        U[(i_offset+0)*m + (j_offset+0)][col] = 1.0
        U[(i_offset+0)*m + (j_offset+1)][col] = 1.0
        V[(j_offset+1)*p + (k_start+1)][col] = 1.0
        W[(k_start+0)*n + (i_offset+0)][col] = 1.0
        W[(k_start+1)*n + (i_offset+0)][col] = 1.0
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[(i_offset+1)*m + (j_offset+0)][col] = 1.0
        U[(i_offset+0)*m + (j_offset+0)][col] = -1.0
        V[(j_offset+0)*p + (k_start+0)][col] = 1.0
        V[(j_offset+0)*p + (k_start+1)][col] = 1.0
        W[(k_start+1)*n + (i_offset+1)][col] = 1.0
        col += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[(i_offset+0)*m + (j_offset+1)][col] = 1.0
        U[(i_offset+1)*m + (j_offset+1)][col] = -1.0
        V[(j_offset+1)*p + (k_start+0)][col] = 1.0
        V[(j_offset+1)*p + (k_start+1)][col] = 1.0
        W[(k_start+1)*n + (i_offset+1)][col] = 1.0
        col += 1
    
    # Block 1: A col 0 (2x1), B row 0 (1x4) for cols 0-3
    add_direct_2x1_x_1x4(0, 0, 0)  # 8 terms, col 0-7
    
    # Block 2: A cols 1-2 (2x2), B rows 1-2 (2x4) for cols 0-3
    # Split into (2x2)x(2x2) for cols 0-1 and (2x2)x(2x2) for cols 2-3
    add_strassen_2x2_x_2x2(0, 1, 0)  # 7 terms, col 8-14
    add_strassen_2x2_x_2x2(0, 1, 2)  # 7 terms, col 15-21
    
    # Block 3: A col 0 (2x1), B row 0 (1x4) for cols 4-7
    add_direct_2x1_x_1x4(0, 0, 4)  # 8 terms, col 22-29
    
    # Block 4: A cols 1-2 (2x2), B rows 1-2 (2x4) for cols 4-7
    # Split into (2x2)x(2x2) for cols 4-5 and (2x2)x(2x2) for cols 6-7
    add_strassen_2x2_x_2x2(0, 1, 4)  # 7 terms, col 30-36
    add_strassen_2x2_x_2x2(0, 1, 6)  # 7 terms, col 37-43
    
    assert col == 44, f"Expected 44 columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
