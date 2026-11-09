"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x5 * 5x5 matrix multiplication.
    
    We partition the 5x5 matrix B into blocks along rows and columns.
    A is 2x5, B is 5x5, result C is 2x5.
    
    C[i, k] = sum_j A[i,j] * B[j,k]
    
    We can group the j-index into blocks. For a 2x2 block of B (rows j1,j2, cols k1,k2),
    the contribution to C[i, k1] and C[i, k2] from rows j1,j2 is:
      C[i,k1] += A[i,j1]*B[j1,k1] + A[i,j2]*B[j2,k1]
      C[i,k2] += A[i,j1]*B[j1,k2] + A[i,j2]*B[j2,k2]
    
    Using Strassen for the 2x2x2 block (A is 2x2 submatrix, B is 2x2, C is 2x2):
    Standard Strassen: 7 multiplications instead of 8.
    
    Partition: 5 = 2 + 2 + 1
    So B has row blocks [0:2], [2:4], [4:5] and col blocks [0:2], [2:4], [4:5].
    A has col blocks [0:2], [2:4], [4:5].
    C has col blocks [0:2], [2:4], [4:5].
    
    For each (row_block_of_B, col_block_of_B) pair, we compute the corresponding
    contribution to C. The A columns corresponding to that row_block are used.
    
    Block sizes:
    - (2,2) block: A_sub is 2x2, B_sub is 2x2, C_sub is 2x2 -> Strassen rank 7
    - (2,1) block: A_sub is 2x2, B_sub is 2x1, C_sub is 2x1 -> schoolbook rank 4 (or can we do better?)
    - (1,2) block: A_sub is 2x1, B_sub is 1x2, C_sub is 2x2 -> schoolbook rank 4
    - (1,1) block: A_sub is 2x1, B_sub is 1x1, C_sub is 2x1 -> schoolbook rank 2
    
    Total blocks: 3x3 = 9 blocks
    (2,2): 4 blocks, each rank 7 -> 28
    (2,1): 2 blocks (row block 2x2 or 2x1, col block 1), each rank 4 -> 8
    (1,2): 2 blocks (row block 1, col block 2x2 or 2x1), each rank 4 -> 8  
    (1,1): 1 block, rank 2 -> 2
    
    Wait, let me recount. Row blocks of B: [0:2](size2), [2:4](size2), [4:5](size1)
    Col blocks of B: [0:2](size2), [2:4](size2), [4:5](size1)
    
    Block (r,c) where r is row block index, c is col block index:
    A columns: same as row block r of B (since A[i,j] multiplies B[j,k])
    
    So A_sub for block (r,c) is A[:, row_block_r] which has size 2 x size(row_block_r)
    B_sub is B[row_block_r, col_block_c] which has size size(row_block_r) x size(col_block_c)
    C_sub is C[:, col_block_c] which has size 2 x size(col_block_c)
    
    Blocks:
    (0,0): A 2x2, B 2x2, C 2x2 -> Strassen, rank 7
    (0,1): A 2x2, B 2x2, C 2x2 -> Strassen, rank 7
    (0,2): A 2x2, B 2x1, C 2x1 -> schoolbook, rank 4
    (1,0): A 2x2, B 2x2, C 2x2 -> Strassen, rank 7
    (1,1): A 2x2, B 2x2, C 2x2 -> Strassen, rank 7
    (1,2): A 2x2, B 2x1, C 2x1 -> schoolbook, rank 4
    (2,0): A 2x1, B 1x2, C 2x2 -> schoolbook, rank 4
    (2,1): A 2x1, B 1x2, C 2x2 -> schoolbook, rank 4
    (2,2): A 2x1, B 1x1, C 2x1 -> schoolbook, rank 2
    
    Total rank: 7*4 + 4*4 + 2 = 28 + 16 + 2 = 46
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Block boundaries
    # Rows of B (and cols of A): [0,2), [2,4), [4,5)
    row_blocks = [(0, 2), (2, 4), (4, 5)]
    # Cols of B (and cols of C): [0,2), [2,4), [4,5)
    col_blocks = [(0, 2), (2, 4), (4, 5)]
    
    U = []  # will be list of lists, shape (n*m, rank)
    V = []  # shape (m*p, rank)
    W = []  # shape (p*n, rank)
    
    current_rank = 0
    
    def add_strassen_2x2x2(A_row_start, A_col_start, B_row_start, B_col_start, C_col_start):
        """
        Strassen decomposition for 2x2 * 2x2 -> 2x2.
        A_sub = A[A_row_start:A_row_start+2, A_col_start:A_col_start+2]  (2x2)
        B_sub = B[B_row_start:B_row_start+2, B_col_start:B_col_start+2]  (2x2)
        C_sub = C[:, C_col_start:C_col_start+2]  (2x2, but only the part from this block)
        
        Strassen:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B01-B11)
        M4 = A11*(B10-B00)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B01)
        M7 = (A01-A11)*(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        nonlocal current_rank
        # A indices: A[i, j] where i in [0,2), j in [A_col_start, A_col_start+2)
        # U index for A[i,j]: i*m + j = i*5 + j
        # V index for B[j,k]: j*p + k = j*5 + k
        # W index for C[i,k]: k*n + i = k*2 + i
        
        # Strassen terms
        # Each term: U_vector * V_vector * W_vector
        # We need to express each M as a product of (linear combo of A) * (linear combo of B)
        # and then C entries as linear combos of M's.
        
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # For each M, we create one rank-1 term
        # U part: coefficients for A entries
        # V part: coefficients for B entries
        # W part: coefficients for C entries (will be filled when we add to C)
        
        # Actually, for tensor decomposition, each rank-1 term is:
        # u (vector in R^{n*m}) * v (vector in R^{m*p}) * w (vector in R^{p*n})
        # The contribution to the tensor is u ⊗ v ⊗ w
        # Which means: sum over i,j,k of u[i,j]*v[j,k]*w[k,i] should match T[i,j,k]
        
        # For Strassen, we have 7 intermediate products, and C is a linear combo of them.
        # This means we need 7 rank-1 terms for the products, but the W vectors encode
        # how each product contributes to each C entry.
        
        # Let me define the 7 Strassen terms:
        # Term t has:
        #   U_t[i*m+j] = coefficient of A[i,j] in the first factor
        #   V_t[j*p+k] = coefficient of B[j,k] in the second factor  
        #   W_t[k*n+i] = coefficient of this term in C[i,k]
        
        # M1: first factor = A00+A11, second factor = B00+B11
        # U: A[0,0]=1, A[1,1]=1 -> indices 0*5+0=0, 1*5+1=6
        # V: B[0,0]=1, B[1,1]=1 -> indices 0*5+0=0, 1*5+1=6
        # W: C[0,0]+=1, C[1,1]+=1 -> indices 0*2+0=0, 1*2+1=3
        # But wait, C00 = M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
        # So W for M1: C[0,0]=1, C[1,1]=1 -> W[0*2+0]=W[0]=1, W[1*2+1]=W[3]=1
        
        strassen_terms = []
        
        # M1: (A00+A11)*(B00+B11)
        # A00: i=0,j=0 -> U idx 0; A11: i=1,j=1 -> U idx 6
        # B00: j=0,k=0 -> V idx 0; B11: j=1,k=1 -> V idx 6
        # C00: i=0,k=0 -> W idx 0; C11: i=1,k=1 -> W idx 3
        strassen_terms.append({
            'U': {0: 1, 6: 1},  # A[0,0] + A[1,1]
            'V': {0: 1, 6: 1},  # B[0,0] + B[1,1]
            'W': {0: 1, 3: 1}   # C[0,0] + C[1,1]
        })
        
        # M2: (A10+A11)*B00
        # A10: i=1,j=0 -> U idx 5; A11: i=1,j=1 -> U idx 6
        # B00: j=0,k=0 -> V idx 0
        # C10: i=1,k=0 -> W idx 2
        strassen_terms.append({
            'U': {5: 1, 6: 1},
            'V': {0: 1},
            'W': {2: 1}
        })
        
        # M3: A00*(B01-B11)
        # A00: i=0,j=0 -> U idx 0
        # B01: j=0,k=1 -> V idx 1; B11: j=1,k=1 -> V idx 6
        # C01: i=0,k=1 -> W idx 1
        strassen_terms.append({
            'U': {0: 1},
            'V': {1: 1, 6: -1},
            'W': {1: 1}
        })
        
        # M4: A11*(B10-B00)
        # A11: i=1,j=1 -> U idx 6
        # B10: j=1,k=0 -> V idx 5; B00: j=0,k=0 -> V idx 0
        # C00: i=0,k=0 -> W idx 0; C10: i=1,k=0 -> W idx 2
        strassen_terms.append({
            'U': {6: 1},
            'V': {5: 1, 0: -1},
            'W': {0: 1, 2: 1}
        })
        
        # M5: (A00+A01)*B11
        # A00: i=0,j=0 -> U idx 0; A01: i=0,j=1 -> U idx 1
        # B11: j=1,k=1 -> V idx 6
        # C00: i=0,k=0 -> W idx 0; C01: i=0,k=1 -> W idx 1
        strassen_terms.append({
            'U': {0: 1, 1: 1},
            'V': {6: 1},
            'W': {0: -1, 1: 1}  # C00 -= M5, C01 += M5
        })
        
        # M6: (A10-A00)*(B00+B01)
        # A10: i=1,j=0 -> U idx 5; A00: i=0,j=0 -> U idx 0
        # B00: j=0,k=0 -> V idx 0; B01: j=0,k=1 -> V idx 1
        # C11: i=1,k=1 -> W idx 3
        strassen_terms.append({
            'U': {5: 1, 0: -1},
            'V': {0: 1, 1: 1},
            'W': {3: 1}
        })
        
        # M7: (A01-A11)*(B10+B11)
        # A01: i=0,j=1 -> U idx 1; A11: i=1,j=1 -> U idx 6
        # B10: j=1,k=0 -> V idx 5; B11: j=1,k=1 -> V idx 6
        # C00: i=0,k=0 -> W idx 0
        strassen_terms.append({
            'U': {1: 1, 6: -1},
            'V': {5: 1, 6: 1},
            'W': {0: 1}
        })
        
        # Now offset the indices
        # A columns are at A_col_start, A_col_start+1
        # B rows are at B_row_start, B_row_start+1
        # B cols are at B_col_start, B_col_start+1
        # C cols are at C_col_start, C_col_start+1
        
        for term in strassen_terms:
            new_U = {}
            for (idx, coeff) in term['U'].items():
                # idx is local: i*2 + j_local, where j_local is 0 or 1
                # Global: i*m + (A_col_start + j_local)
                i_local = idx // 2
                j_local = idx % 2
                global_idx = i_local * m + (A_col_start + j_local)
                new_U[global_idx] = coeff
            term['U'] = new_U
            
            new_V = {}
            for (idx, coeff) in term['V'].items():
                # idx is local: j_local*2 + k_local
                # Global: (B_row_start + j_local)*p + (B_col_start + k_local)
                j_local = idx // 2
                k_local = idx % 2
                global_idx = (B_row_start + j_local) * p + (B_col_start + k_local)
                new_V[global_idx] = coeff
            term['V'] = new_V
            
            new_W = {}
            for (idx, coeff) in term['W'].items():
                # idx is local: k_local*2 + i_local
                # Global: (C_col_start + k_local)*n + i_local
                k_local = idx // 2
                i_local = idx % 2
                global_idx = (C_col_start + k_local) * n + i_local
                new_W[global_idx] = coeff
            term['W'] = new_W
        
        # Add to U, V, W
        for term in strassen_terms:
            U.append([0] * (n * m))
            V.append([0] * (m * p))
            W.append([0] * (p * n))
            rank_idx = current_rank
            for idx, coeff in term['U'].items():
                U[rank_idx][idx] = coeff
            for idx, coeff in term['V'].items():
                V[rank_idx][idx] = coeff
            for idx, coeff in term['W'].items():
                W[rank_idx][idx] = coeff
            current_rank += 1
    
    def add_schoolbook_block(A_row_start, A_col_start, A_rows, A_cols, B_row_start, B_col_start, B_rows, B_cols, C_col_start):
        """
        Schoolbook for A[A_row_start:A_row_start+A_rows, A_col_start:A_col_start+A_cols]
        * B[B_row_start:B_row_start+B_rows, B_col_start:B_col_start+B_cols]
        -> C[:, C_col_start:C_col_start+B_cols]
        
        A_sub is A_rows x A_cols (A_rows should be 2 for our case)
        B_sub is B_rows x B_cols
        C_sub is 2 x B_cols
        """
        nonlocal current_rank
        for i in range(A_rows):  # i is 0 or 1 (row of A and C)
            for j_local in range(A_cols):  # j_local is column of A_sub, row of B_sub
                for k_local in range(B_cols):  # k_local is column of B_sub, column of C_sub
                    # A[i, A_col_start+j_local] * B[B_row_start+j_local, B_col_start+k_local]
                    # contributes to C[i, C_col_start+k_local]
                    
                    U.append([0] * (n * m))
                    V.append([0] * (m * p))
                    W.append([0] * (p * n))
                    rank_idx = current_rank
                    
                    U[rank_idx][(A_row_start + i) * m + (A_col_start + j_local)] = 1
                    V[rank_idx][(B_row_start + j_local) * p + (B_col_start + k_local)] = 1
                    W[rank_idx][(C_col_start + k_local) * n + (A_row_start + i)] = 1
                    
                    current_rank += 1
    
    # Process each block
    for r_idx, (r_start, r_end) in enumerate(row_blocks):
        for c_idx, (c_start, c_end) in enumerate(col_blocks):
            A_rows = 2  # A always has 2 rows
            A_cols = r_end - r_start  # columns of A corresponding to this row block of B
            B_rows = r_end - r_start
            B_cols = c_end - c_start
            
            if A_cols == 2 and B_rows == 2 and B_cols == 2:
                add_strassen_2x2x2(0, r_start, r_start, c_start, c_start)
            else:
                add_schoolbook_block(0, r_start, A_rows, A_cols, r_start, c_start, B_rows, B_cols, c_start)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
