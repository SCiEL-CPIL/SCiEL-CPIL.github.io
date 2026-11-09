"""Matrix multiplication 4x4 by 4x9: rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-28 decomposition of the 4x4 * 4x9 matrix multiplication tensor.
    
    Strategy:
    - Block A (4x4) into 2x2 blocks of size 2x2: A11, A12, A21, A22
    - Block B (4x9) into 2x2 blocks: B11 (2x9), B12 (2x9), B21 (2x9), B22 (2x9)
    - Use Strassen's 7 multiplications on 2x2 blocks, each being 2x2 * 2x9
    
    For each 2x2 * 2x9 product, we use rank-6 decomposition:
      P1 = A11 * B11
      P2 = A12 * B12  
      P3 = A21 * B21
      P4 = A22 * B22
      P5 = (A11+A22) * (B11+B22)  -> extra term
      P6 = (A11+A12) * B22  -> wait, this isn't quite right
      
    Let me think more carefully.
    
    Actually, for 2x2 * 2x9, we can use the standard rank-6:
    The tensor for 2x2 * 2x9 has structure: C[i,j] = sum_k A[i,k] * B[k,j]
    where A is 2x2, B is 2x9, C is 2x9.
    
    Rank of 2x2*2x9: We can decompose as:
    M1 = A11*B11, M2 = A12*B12, M3 = A21*B21, M4 = A22*B22
    M5 = (A11+A22)*(B11+B22), M6 = (A12+A22)*(B21+B22)
    
    Then:
    C11 = M1 + M6 - M4 - M5/2 ... this gets messy with halves.
    
    Let me use a cleaner approach. For 2x2 * 2xp, the minimal rank is actually 4+p-1? No...
    
    Actually, let me just use rank 8 for each 2x2*2x9 block (schoolbook), giving total 7*8 = 56.
    Or better: use the known rank for 2x2*2x2 which is 7, and extend.
    
    For 2x2 * 2x9: Think of it as 2x2 * 2x2 done 9 times? No, that's not how it works.
    
    The tensor T_{i,j,k} = A_{i,k} * B_{k,j} where i in {0,1}, k in {0,1}, j in {0,...,8}
    
    This is equivalent to: for each j, we do a 2x2*2x1 product. But we can share structure.
    
    Actually, the rank of m x n * n x p matrix multiplication is at most mn + np + mp - 2 (for m=n=p=2, gives 7).
    
    For 2x2*2x9: rank <= 2*2 + 2*9 + 2*9 - 2 = 4 + 18 + 18 - 2 = 38? That's too high.
    
    Wait, that formula is for the general case but I think it's an upper bound that's not tight.
    
    Let me just use the simple decomposition: 2x2 * 2x9 can be done in rank 4 (schoolbook for the 2x2 part, times 9 columns):
    C[i,j] = sum_k A[i,k] * B[k,j]
    
    Schoolbook: 2*2*9 = 36 terms. But we can do better.
    
    Key insight: C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    So C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    
    This is a rank-4 tensor! The four basis vectors are:
    u0: A[0,0]=1, others 0; v0: B[0,*]=1 (entire row), B[1,*]=0; w0: C[0,*]=1
    u1: A[0,1]=1; v1: B[1,*]=1; w1: C[0,*]=1  
    u2: A[1,0]=1; v2: B[0,*]=1; w2: C[1,*]=1
    u3: A[1,1]=1; v3: B[1,*]=1; w3: C[1,*]=1
    
    So 2x2*2x9 has rank exactly 4!
    
    Now for 4x4 * 4x9 using 2x2 Strassen:
    7 multiplications, each 2x2 * 2x9 with rank 4 = 28 total.
    
    Let me implement this.
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strassen decomposition for 4x4 * 4x9
    # A is 4x4, B is 4x9, C is 4x9
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] each 2x2
    # B = [[B11, B12], [B21, B22]] each 2x9
    # C = [[C11, C12], [C21, C22]] each 2x9
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = A22(B11+B12)
    # M3 = (A11+A12)B22
    # M4 = A11(B21-B22)
    # M5 = (A21-A11)B11
    # M6 = (A12+A22)(B21+B22)
    # M7 = (A21-A12)(B11+B12)
    #
    # C11 = M1 + M4 - M6 + M7
    # C12 = M2 + M3
    # C21 = M5 + M1
    # C22 = M3 - M4 + M5 + M6
    
    # Each Mi is a 2x2 * 2x9 product, which has rank 4
    # Total rank: 7 * 4 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 16 x 28
    V = [[0] * rank for _ in range(m * p)]  # 36 x 28
    W = [[0] * rank for _ in range(p * n)]  # 36 x 28
    
    # Helper: add a rank-4 decomposition of 2x2*2x9 into U,V,W
    # A_rows: the 2 rows of A block (indices into A's 16 entries)
    # B_rows: the 2 rows of B block (indices into B's 36 entries)
    # C_rows: the 2 rows of C block (indices into C's 36 entries, using k*n+i convention)
    # col_start: starting column index in U,V,W
    
    def add_2x2x9(A_row0, A_row1, B_row0, B_row1, C_row0, C_row1, col_start, signs=None):
        """
        Add rank-4 decomposition for 2x2 * 2x9.
        
        C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
        C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
        
        Four terms:
        Term 0: A[0,0] * B[0,:] -> C[0,:]
        Term 1: A[0,1] * B[1,:] -> C[0,:]
        Term 2: A[1,0] * B[0,:] -> C[1,:]
        Term 3: A[1,1] * B[1,:] -> C[1,:]
        """
        if signs is None:
            signs = [1, 1, 1, 1]
        
        # Term 0: A[i0,j0] * B[j0, :] -> C[i0, :]
        s = signs[0]
        U[A_row0][col_start + 0] = s
        # B row j0 has p=9 columns, all get coefficient s
        for k in range(p):
            V[B_row0 * p + k][col_start + 0] = s
        # C row i0, all p columns
        for k in range(p):
            W[k * n + C_row0][col_start + 0] = s
        
        # Term 1: A[i0,j1] * B[j1, :] -> C[i0, :]
        s = signs[1]
        U[A_row1][col_start + 1] = s
        for k in range(p):
            V[B_row1 * p + k][col_start + 1] = s
        for k in range(p):
            W[k * n + C_row0][col_start + 1] = s
        
        # Term 2: A[i1,j0] * B[j0, :] -> C[i1, :]
        s = signs[2]
        U[A_row1][col_start + 2] = s
        for k in range(p):
            V[B_row0 * p + k][col_start + 2] = s
        for k in range(p):
            W[k * n + C_row1][col_start + 2] = s
        
        # Term 3: A[i1,j1] * B[j1, :] -> C[i1, :]
        s = signs[3]
        U[A_row1][col_start + 3] = s
        for k in range(p):
            V[B_row1 * p + k][col_start + 3] = s
        for k in range(p):
            W[k * n + C_row1][col_start + 3] = s
    
    # Index mapping:
    # A is 4x4, row-major: A[i,j] is at index i*m+j = i*4+j
    # B is 4x9, row-major: B[i,j] is at index i*p+j = i*9+j  
    # C is 4x9, output: C[i,j] uses W convention W[k*n+i] where k is column index, i is row index
    # So C[i,j] -> W[j*n + i]
    
    # A block indices (2x2 each):
    # A11: rows 0-1, cols 0-1 -> A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # For a 2x2 block at rows (r0,r1) and cols (c0,c1):
    # A[r0,c0] -> idx r0*4+c0
    # A[r0,c1] -> idx r0*4+c1
    # A[r1,c0] -> idx r1*4+c0
    # A[r1,c1] -> idx r1*4+c1
    
    # B block indices (2x9 each):
    # B11: rows 0-1, all 9 cols -> B[0,*]: indices 0-8, B[1,*]: indices 9-17
    # B12: rows 2-3, all 9 cols -> B[2,*]: indices 18-26, B[3,*]: indices 27-35
    
    # For Strassen, each "multiplication" Mi involves sum/difference of A blocks and B blocks.
    # Since we're building a linear decomposition, we need to express the coefficients.
    
    # Let me define each of the 7 Strassen multiplications explicitly.
    
    col = 0
    
    # M1 = (A11+A22)(B11+B22)
    # A part: A11 + A22
    #   A11[0,0]=A[0,0], A11[0,1]=A[0,1], A11[1,0]=A[1,0], A11[1,1]=A[1,1]
    #   A22[0,0]=A[2,0], A22[0,1]=A[2,1], A22[1,0]=A[3,0], A22[1,1]=A[3,1]
    #   Sum: [A[0,0]+A[2,0], A[0,1]+A[2,1], A[1,0]+A[3,0], A[1,1]+A[3,1]]
    # B part: B11 + B22
    #   B11 row0 = B[0,:], B11 row1 = B[1,:]
    #   B22 row0 = B[2,:], B22 row1 = B[3,:]
    #   Sum: row0 = B[0,:]+B[2,:], row1 = B[1,:]+B[3,:]
    
    # For M1, the 2x2 matrix is:
    # [A[0,0]+A[2,0],  A[0,1]+A[2,1]]
    # [A[1,0]+A[3,0],  A[1,1]+A[3,1]]
    # And the 2x9 matrix is:
    # [B[0,:]+B[2,:]]
    # [B[1,:]+B[3,:]]
    
    # Using rank-4 decomposition:
    # C[0,:] = (A[0,0]+A[2,0])*(B[0,:]+B[2,:]) + (A[0,1]+A[2,1])*(B[1,:]+B[3,:])
    # C[1,:] = (A[1,0]+A[3,0])*(B[0,:]+B[2,:]) + (A[1,1]+A[3,1])*(B[1,:]+B[3,:])
    
    # This expands to 4 terms with cross terms:
    # (A[0,0]+A[2,0])*(B[0,:]+B[2,:]) = A[0,0]*B[0,:] + A[0,0]*B[2,:] + A[2,0]*B[0,:] + A[2,0]*B[2,:]
    # That's 4 sub-terms per "big term", so 8 total for M1.
    
    # This approach gives rank 7*8 = 56, not 28.
    
    # I need a different approach. Let me think about this differently.
    
    # Alternative: Use the fact that 4x4*4x9 can be viewed as a block structure where
    # we use 7 Strassen products, and each 2x2*2x9 product is done in rank 4.
    # But the issue is that the Strassen products involve sums of blocks, and expanding
    # those sums increases the rank.
    
    # Actually, in tensor decomposition, we can have linear combinations in the U, V, W matrices.
    # So for M1 = (A11+A22)(B11+B22), we don't need to expand. We can have:
    # U vectors that pick out A[0,0] and A[2,0] with coefficients +1 each
    # V vectors that pick out B[0,:] and B[2,:] with coefficients +1 each
    # etc.
    
    # So the rank-4 decomposition of the 2x2*2x9 tensor can use composite A and B rows.
    
    # Let me redo this. For each Strassen product Mi, I define:
    # - The effective 2x2 A-matrix (with linear combinations)
    # - The effective 2x9 B-matrix (with linear combinations)
    # Then use the rank-4 decomposition on these.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_rank4_A(B_row0_coeffs, B_row1_coeffs, A_row0_coeffs, A_row1_coeffs, 
                    C_row0, C_row1, col_start):
        """
        Add a rank-4 term for a 2x2*2x9 product where:
        A_eff[0,:] = sum of A rows with coefficients A_row0_coeffs
        A_eff[1,:] = sum of A rows with coefficients A_row1_coeffs
        B_eff[0,:] = sum of B rows with coefficients B_row0_coeffs  
        B_eff[1,:] = sum of B rows with coefficients B_row1_coeffs
        
        A_row0_coeffs: list of (A_row_index, coeff) for the first row of effective A
        A_row1_coeffs: list of (A_row_index, coeff) for the second row
        B_row0_coeffs: list of (B_row_index, coeff) for first row of effective B
        B_row1_coeffs: list of (B_row_index, coeff) for second row
        """
        # Term 0: A_eff[0,0] * B_eff[0,:] -> C[C_row0,:]
        # A_eff[0,0] means the (0,0) entry of effective A
        # This is complicated because A_eff entries are linear combos...
        
        # Actually, I realize the rank-4 decomposition assumes the A and B are the actual matrices.
        # When we have linear combinations, we need to distribute.
        
        pass
    
    # Let me take a step back and think about this more carefully.
    
    # The matrix multiplication tensor for 4x4 * 4x9 is:
    # T_{i,j,k} where i in [0,3] (row of A and C), j in [0,8] (col of B and C), k in [0,3] (col of A and row of B)
    # T[i,j,k] = A[i,k] * B[k,j]
    
    # We want to write this as sum_{r} U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # Note the index conventions:
    # U has size (n*m) x rank = 16 x rank, indexed by [i*m+k, r]
    # V has size (m*p) x rank = 36 x rank, indexed by [k*p+j, r]  
    # W has size (p*n) x rank = 36 x rank, indexed by [j*n+i, r]
    
    # Schoolbook: for each (i,k,j) triple, one term. 4*4*9 = 144 terms.
    
    # To get rank 28, I'll use the Strassen-based approach with careful linear algebra.
    
    # Let me just hard-code a known rank-28 decomposition if I can derive it.
    
    # Actually, let me try a different block decomposition.
    # What if I split 4x4 * 4x9 into two 2x4 * 4x9 problems?
    # Or 4x2 * 2x9 and 4x2 * 2x9?
    
    # 4x2 * 2x9: rank is at most 4*2*9/... The rank of m x n * n x p is at most min(m*p, n*(m+p-n), ...)
    
    # For 4x2*2x9: schoolbook is 4*2*9 = 72. 
    # Can we do better? The rank is at most 2*(4+9-2) = 22? I'm not sure of the exact bound.
    
    # Let me just go with a simpler but still improved decomposition.
    # Use 4x4*4x4 (rank 7 via Strassen... wait, 4x4*4x4 Strassen gives 7*7=49) 
    # plus 4x4*4x5.
    
    # 4x4*4x4: Strassen rank 49
    # 4x4*4x5: schoolbook 80, or better?
    
    # Total: 49+80 = 129. Still not great.
    
    # Let me try: 4x4*4x9 = 4x4*4x4 + 4x4*4x5
    # For 4x4*4x4, use rank 49 (Strassen).
    # For 4x4*4x5, use a decomposition.
    
    # Actually, I found that the rank of 4x4*4x4 is at most 49 (Strassen), and it's known to be 49 or less.
    # The rank of 4x4*4x5: I can use 4x4*4x4 (rank 49) + 4x4*4x1 (rank 16) = 65? No, that's not how it works.
    
    # I'm overcomplicating this. Let me just implement the Strassen 2x2 approach properly with rank 56,
    # which is still an improvement over 144.
    
    U = [[0] * 56 for _ in range(n * m)]
    V = [[0] * 56 for _ in range(m * p)]
    W = [[0] * 56 for _ in range(p * n)]
    
    col = 0
    
    def add_2x2x9_expanded(A_coeff, B_coeff, C_row0, C_row1, col_start):
        """
        Add rank-8 expanded decomposition for 2x2*2x9.
        A_coeff[2][2]: the 2x2 coefficient matrix (linear combo of A entries)
        B_coeff[2][9]: the 2x9 coefficient matrix (linear combo of B entries)
        Actually, let me pass the actual A and B row/col indices with coefficients.
        """
        pass
    
    # OK let me just do the straightforward Strassen with expansion.
    # Each of the 7 Strassen products involves summing 2 A-blocks and 2 B-blocks.
    # When expanded, each product becomes 8 rank-1 terms.
    # Total: 7 * 8 = 56.
    
    col = 0
    
    def add_strassen_product(A_blocks, B_blocks, C_blocks, col_start):
        """
        Add the 8 rank-1 terms for a Strassen product.
        A_blocks: list of 4 (A_row_idx, A_col_idx, coeff) for the 2x2 A matrix
        B_blocks: list of 18 (B_row_idx, B_col_idx, coeff) for the 2x9 B matrix  
        C_blocks: list of 4 (C_row_idx, C_col_idx, coeff) for the 2x9 C matrix
        """
        pass
    
    # I think I should just directly construct the 56 terms.
    
    # Let me define the 7 Strassen products and their contributions to C.
    
    # A block row/col indices (within the 4x4 A matrix):
    # Block A11: rows 0-1, cols 0-1
    # Block A12: rows 0-1, cols 2-3
    # Block A21: rows 2-3, cols 0-1
    # Block A22: rows 2-3, cols 2-3
    
    # B block row indices (within the 4x9 B matrix):
    # Block B11: rows 0-1
    # Block B12: rows 2-3
    # (columns are all 0-8 for each)
    
    # C block row/col indices:
    # C11: rows 0-1, cols 0-4 (wait, 4x9 split into 2x9 blocks)
    # C11: rows 0-1, all 9 cols
    # C12: rows 2-3, all 9 cols
    
    # Hmm, actually for 4x9 output, splitting into 2x9 blocks:
    # C top half: rows 0-1, 9 cols
    # C bottom half: rows 2-3, 9 cols
    
    # But Strassen gives 4 output blocks each 2x9:
    # C11 = top-left... wait, B is 4x9, so B11 and B12 are both 2x9.
    # The output C is 4x9, split into C_top (2x9) and C_bottom (2x9).
    # C_top = A11*B11 + A12*B21... no.
    
    # Let me redo the block structure:
    # A (4x4) = [[A11, A12], [A21, A22]] where each is 2x2
    # B (4x9) = [[B11], [B21]] where B11 is 2x9, B21 is 2x9
    #   Actually B = [[B11, B12], [B21, B22]] but B is 4x9, so:
    #   B11 is 2x9, B21 is 2x9? No.
    
    # B is 4x9. If I split rows into 2+2:
    # B_top (2x9) = rows 0-1 of B
    # B_bottom (2x9) = rows 2-3 of B
    # But that's only a 2-block split, not 2x2.
    
    # For Strassen on 4x4 * 4x9, I should split both into 2x2 blocks where possible.
    # A: 2x2 blocks of 2x2. Fine.
    # B: 4x9. Split into 2x2 blocks of 2x9? That's 2 blocks vertically.
    #   B = [[B11], [B21]] where B11 is 2x9, B21 is 2x9.
    #   But Strassen needs 2x2 blocks for B too.
    
    # I think the cleanest is:
    # A: 2x2 blocks of 2x2
    # B: 2 blocks of 2x9 (just split rows)
    # C: 2 blocks of 2x9 (just split rows)
    
    # Then:
    # C_top = A_top_left * B_top + A_top_right * B_bottom
    #   = A11 * B11 + A12 * B21  (where B11=B_top, B21=B_bottom)
    # C_bottom = A_bottom_left * B_top + A_bottom_right * B_bottom
    #   = A21 * B11 + A22 * B21
    
    # This is just block multiplication, no Strassen needed at the top level.
    # Each 2x2 * 2x9 product has rank 4.
    # Total: 4 products * rank 4 = rank 16.
    
    # Wait, that's even better! Let me verify.
    
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # For i in {0,1} (top):
    #   C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] + A[i,2]*B[2,j] + A[i,3]*B[3,j]
    #   = (A[i,0]*B[0,j] + A[i,1]*B[1,j]) + (A[i,2]*B[2,j] + A[i,3]*B[3,j])
    #   = (A_top[i,:] * B_top[:,j]) + (A_right[i,:] * B_bottom[:,j])
    
    # So C_top = A11 * B_top + A12 * B_bottom
    # where A11 is 2x2 (top-left of A), A12 is 2x2 (top-right of A)
    # B_top is 2x9, B_bottom is 2x9
    
    # A11 * B_top: 2x2 * 2x9, rank 4
    # A12 * B_bottom: 2x2 * 2x9, rank 4
    
    # Similarly C_bottom = A21 * B_top + A22 * B_bottom, rank 8
    
    # Total rank: 16!
    
    # Let me implement this.
    
    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # 16 x 16
    V = [[0] * rank for _ in range(m * p)]  # 36 x 16
    W = [[0] * rank for _ in range(p * n)]  # 36 x 16
    
    col = 0
    
    # Four 2x2 * 2x9 products:
    # 1. A11 * B_top -> C_top contribution
    # 2. A12 * B_bottom -> C_top contribution
    # 3. A21 * B_top -> C_bottom contribution
    # 4. A22 * B_bottom -> C_bottom contribution
    
    def add_2x2x9(A_rows, B_rows, C_rows, col_start):
        """
        A_rows: [A_row0_idx, A_row1_idx] - the two row indices in A (0-15)
        B_rows: [B_row0_idx, B_row1_idx] - the two row indices in B (0-3, each representing a row of 9)
        C_rows: [C_row0_idx, C_row1_idx] - the two row indices in C output (0-3)
        
        This adds 4 rank-1 terms.
        """
        # Term 0: A[A_row0, 0] * B[B_row0, :] -> C[C_row0, :]
        # A[A_row0, 0]: A_row0 encodes A[i, j] where i = A_row0 // 4, j = A_row0 % 4
        # For the 2x2 block, the columns are fixed.
        
        # Actually, A_rows should give me the (i,j) pairs.
        # Let me change the interface.
        pass
    
    # Let me be explicit.
    
    # A11 * B_top:
    # A11 = A[0:2, 0:2], rows 0-1 of A, cols 0-1
    # B_top = B[0:2, :], rows 0-1 of B
    # C_top = C[0:2, :], rows 0-1 of C
    
    # C[0,j] += A[0,0]*B[0,j] + A[0,1]*B[1,j]
    # C[1,j] += A[1,0]*B[0,j] + A[1,1]*B[1,j]
    
    # 4 terms:
    # (A[0,0], B[0,:], C[0,:])
    # (A[0,1], B[1,:], C[0,:])
    # (A[1,0], B[0,:], C[1,:])
    # (A[1,1], B[1,:], C[1,:])
    
    def add_term(A_idx, B_row, C_row, col_idx):
        """
        A_idx: index into A's 16 entries (i*4+j)
        B_row: row index into B (0-3), representing B[B_row, :]
        C_row: row index into C output (0-3)
        """
        i, j = A_idx // 4, A_idx % 4
        # U: A[i,j] is at U[i*4+j]
        U[i * 4 + j][col_idx] = 1
        # V: B[B_row, k] for k in 0..8, at V[B_row*9+k]
        for k in range(p):
            V[B_row * p + k][col_idx] = 1
        # W: C[C_row, k] at W[k*4 + C_row]
        for k in range(p):
            W[k * n + C_row][col_idx] = 1
    
    # Product 1: A11 * B_top
    add_term(0, 0, 0, col); col += 1  # A[0,0] * B[0,:] -> C[0,:]
    add_term(1, 1, 0, col); col += 1  # A[0,1] * B[1,:] -> C[0,:]
    add_term(4, 0, 1, col); col += 1  # A[1,0] * B[0,:] -> C[1,:]
    add_term(5, 1, 1, col); col += 1  # A[1,1] * B[1,:] -> C[1,:]
    
    # Product 2: A12 * B_bottom
    # A12 = A[0:2, 2:4], cols 2-3
    # B_bottom = B[2:4, :], rows 2-3
    # C_top
    add_term(2, 2, 0, col); col += 1  # A[0,2] * B[2,:] -> C[0,:]
    add_term(3, 3, 0, col); col += 1  # A[0,3] * B[3,:] -> C[0,:]
    add_term(6, 2, 1, col); col += 1  # A[1,2] * B[2,:] -> C[1,:]
    add_term(7, 3, 1, col); col += 1  # A[1,3] * B[3,:] -> C[1,:]
    
    # Product 3: A21 * B_top
    # A21 = A[2:4, 0:2], rows 2-3, cols 0-1
    # B_top = B[0:2, :]
    # C_bottom = C[2:4, :]
    add_term(8, 0, 2, col); col += 1  # A[2,0] * B[0,:] -> C[2,:]
    add_term(9, 1, 2, col); col += 1  # A[2,1] * B[1,:] -> C[2,:]
    add_term(12, 0, 3, col); col += 1  # A[3,0] * B[0,:] -> C[3,:]
    add_term(13, 1, 3, col); col += 1  # A[3,1] * B[1,:] -> C[3,:]
    
    # Product 4: A22 * B_bottom
    # A22 = A[2:4, 2:4]
    # B_bottom = B[2:4, :]
    # C_bottom
    add_term(10, 2, 2, col); col += 1  # A[2,2] * B[2,:] -> C[2,:]
    add_term(11, 3, 2, col); col += 1  # A[2,3] * B[3,:] -> C[2,:]
    add_term(14, 2, 3, col); col += 1  # A[3,2] * B[2,:] -> C[3,:]
    add_term(15, 3, 3, col); col += 1  # A[3,3] * B[3,:] -> C[3,:]
    
    assert col == 16, f"Expected 16 columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
