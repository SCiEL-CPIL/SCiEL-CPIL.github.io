def construct():
    n, m, p = 2, 4, 8
    # We decompose into 8 independent 2x2x2 multiplications
    # Each uses Strassen with rank 7
    # Total rank = 8 * 7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen 2x2x2 decomposition patterns
    # For a 2x2 A matrix with elements a11,a12,a21,a22
    # and 2x2 B matrix with elements b11,b12,b21,b22
    # The 7 Strassen products and their contributions
    
    # Let's define the Strassen factors for one 2x2x2 block
    # A indices: (0,0),(0,1),(1,0),(1,1) -> row in A part
    # B indices: (0,0),(0,1),(1,0),(1,1) -> col in B part  
    # C indices: (0,0),(0,1),(1,0),(1,1) -> row in C part
    
    # Strassen products (using 0-indexed for 2x2):
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each of the 8 blocks, we assign 7 consecutive columns
    
    for block in range(8):
        base = block * 7
        # Which A rows and B cols?
        # block = i * 4 + k where i in {0,1} (which half of A) and k in {0,1,2,3} (which 2-col block of B)
        i = block // 4  # 0 or 1
        k = block % 4    # 0,1,2,3
        
        # A part: rows i*2 and i*2+1 (since A is 2x4, row i*2 corresponds to the i-th 2-row block)
        # Actually A is 2x4, so A has 2 rows. The "i" here refers to which 2x2 subblock of B we're multiplying
        # Wait, let me re-think the indexing.
        
        # A is 2x4. We split the 4 columns into two groups of 2: cols 0-1 (group 0) and cols 2-3 (group 1)
        # B is 4x8. We split the 4 rows into two groups of 2: rows 0-1 (group 0) and rows 2-3 (group 1)
        # And the 8 cols into four groups of 2: cols 0-1, 2-3, 4-5, 6-7
        
        # For block = group_a * 4 + group_b_col:
        # group_a in {0,1}: which 2x2 subblock of A (columns 2*group_a to 2*group_a+1)
        # group_b_col in {0,1,2,3}: which 2x2 subblock of B (rows 2*group_a to 2*group_a+1, cols 2*group_b_col to 2*group_b_col+1)
        
        # Wait, that's not right. Let me re-examine.
        # A is 2x4: rows 0-1, cols 0-3
        # B is 4x8: rows 0-3, cols 0-7
        # C = A*B is 2x8: rows 0-1, cols 0-7
        
        # C[i][c] = sum_j A[i][j]*B[j][c]
        # = sum_{j=0}^{3} A[i][j]*B[j][c]
        
        # Split j into two groups: j in {0,1} and j in {2,3}
        # C[i][c] = sum_{j=0}^{1} A[i][j]*B[j][c] + sum_{j=2}^{3} A[i][j]*B[j][c]
        
        # For the first sum, define A_left[i][j] = A[i][j] for j=0,1 (2x2)
        # B_top[c_group][j] = B[j][c] for j=0,1 (2x2 for each c_group)
        # Similarly for right/bottom
        
        # So we have:
        # Term1: A_left (2x2) * B_top_block (2x2) for each of 4 col groups
        # Term2: A_right (2x2) * B_bottom_block (2x2) for each of 4 col groups
        
        # That's 8 terms total, each 2x2x2.
        
        # For block index:
        # block = 0..3: A_left * B_top_colgroup[block]
        # block = 4..7: A_right * B_bottom_colgroup[block-4]
        
        if block < 4:
            # A_left: A rows 0-1, cols 0-1
            a_row_base = 0
            a_col_base = 0
            b_row_base = 0  # B rows 0-1
            b_col_base = block * 2
        else:
            # A_right: A rows 0-1, cols 2-3
            a_row_base = 0
            a_col_base = 2
            b_row_base = 2  # B rows 2-3
            b_col_base = (block - 4) * 2
        
        # Now for this 2x2x2 block, assign Strassen factors
        # A local indices: (0,0),(0,1),(1,0),(1,1) correspond to:
        #   A[a_row_base + dr][a_col_base + dc] for dr,dc in {0,1}
        # B local indices: (0,0),(0,1),(1,0),(1,1) correspond to:
        #   B[b_row_base + dr][b_col_base + dc] for dr,dc in {0,1}
        # C local indices: (0,0),(0,1),(1,0),(1,1) correspond to:
        #   C[a_row_base + dr][b_col_base + dc] for dr,dc in {0,1}
        
        # W index: p*n = 16, W[k*n + i, col] where k is col index of C (0-7), i is row index of C (0-1)
        # Wait, the problem says W[k*n+i, r]. Let me re-read.
        # "output factor follows the official trace convention W[k*n+i,r]"
        # So W has shape (p*n, rank) = (16, rank)
        # W[k*n + i, r] where k is output col (0 to p-1=7), i is output row (0 to n-1=1)
        # So W index = k*2 + i
        
        # U index: U[i*m + j, r] where i is A row (0-1), j is A col (0-3)
        # So U index = i*4 + j
        
        # V index: V[j*p + k, r] where j is B row (0-3), k is B col (0-7)  
        # So V index = j*8 + k
        
        # Strassen M1 = (a00+a11)*(b00+b11)
        # a00 = A[a_row_base][a_col_base], a11 = A[a_row_base+1][a_col_base+1]
        # b00 = B[b_row_base][b_col_base], b11 = B[b_row_base+1][b_col_base+1]
        
        u_a00 = a_row_base * m + a_col_base
        u_a11 = (a_row_base + 1) * m + (a_col_base + 1)
        v_b00 = b_row_base * p + b_col_base
        v_b11 = (b_row_base + 1) * p + (b_col_base + 1)
        
        U[u_a00][base + 0] += 1
        U[u_a11][base + 0] += 1
        V[v_b00][base + 0] += 1
        V[v_b11][base + 0] += 1
        
        # C contributions from M1: C00 += 1, C11 += 1
        c00 = b_col_base * n + a_row_base  # W index: k=b_col_base, i=a_row_base
        c11 = (b_col_base + 1) * n + (a_row_base + 1)
        W[c00][base + 0] += 1
        W[c11][base + 0] += 1
        
        # M2 = (a10+a11)*b00
        u_a10 = (a_row_base + 1) * m + a_col_base
        u_a11_idx = (a_row_base + 1) * m + (a_col_base + 1)
        U[u_a10][base + 1] += 1
        U[u_a11_idx][base + 1] += 1
        V[v_b00][base + 1] += 1
        
        # C10 += 1 (from C10 = M2 + M4)
        c10 = b_col_base * n + (a_row_base + 1)
        W[c10][base + 1] += 1
        # C11 -= 1 (from C11 = M1 - M2 + M3 + M6)
        W[c11][base + 1] -= 1
        
        # M3 = a00*(b10-b11)
        u_a00_idx = a_row_base * m + a_col_base
        v_b10 = (b_row_base + 1) * p + b_col_base
        v_b11_idx = (b_row_base + 1) * p + (b_col_base + 1)
        U[u_a00_idx][base + 2] += 1
        V[v_b10][base + 2] += 1
        V[v_b11_idx][base + 2] -= 1
        
        # C01 += 1 (from C01 = M3 + M5)
        c01 = (b_col_base + 1) * n + a_row_base
        W[c01][base + 2] += 1
        # C11 += 1
        W[c11][base + 2] += 1
        
        # M4 = a11*(b01-b00)
        u_a11_idx2 = (a_row_base + 1) * m + (a_col_base + 1)
        v_b01 = b_row_base * p + (b_col_base + 1)
        v_b00_idx = b_row_base * p + b_col_base
        U[u_a11_idx2][base + 3] += 1
        V[v_b01][base + 3] += 1
        V[v_b00_idx][base + 3] -= 1
        
        # C00 += 1
        W[c00][base + 3] += 1
        # C10 += 1
        W[c10][base + 3] += 1
        
        # M5 = (a00+a01)*b11
        u_a01 = a_row_base * m + (a_col_base + 1)
        v_b11_idx2 = (b_row_base + 1) * p + (b_col_base + 1)
        U[u_a00_idx][base + 4] += 1  # a00
        U[u_a01][base + 4] += 1      # a01
        V[v_b11_idx2][base + 4] += 1
        
        # C00 -= 1
        W[c00][base + 4] -= 1
        # C01 += 1
        W[c01][base + 4] += 1
        
        # M6 = (a10-a00)*(b00+b01)
        u_a10_idx = (a_row_base + 1) * m + a_col_base
        u_a00_idx2 = a_row_base * m + a_col_base
        v_b00_idx2 = b_row_base * p + b_col_base
        v_b01_idx = b_row_base * p + (b_col_base + 1)
        U[u_a10_idx][base + 5] += 1   # a10
        U[u_a00_idx2][base + 5] -= 1  # -a00
        V[v_b00_idx2][base + 5] += 1  # b00
        V[v_b01_idx][base + 5] += 1   # b01
        
        # C11 += 1
        W[c11][base + 5] += 1
        
        # M7 = (a01-a11)*(b10+b11)
        u_a01_idx = a_row_base * m + (a_col_base + 1)
        u_a11_idx3 = (a_row_base + 1) * m + (a_col_base + 1)
        v_b10_idx = (b_row_base + 1) * p + b_col_base
        v_b11_idx3 = (b_row_base + 1) * p + (b_col_base + 1)
        U[u_a01_idx][base + 6] += 1   # a01
        U[u_a11_idx3][base + 6] -= 1  # -a11
        V[v_b10_idx][base + 6] += 1   # b10
        V[v_b11_idx3][base + 6] += 1  # b11
        
        # C00 += 1
        W[c00][base + 6] += 1
    
    return {"U": U, "V": V, "W": W}
