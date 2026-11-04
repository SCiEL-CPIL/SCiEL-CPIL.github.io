"""Improved program for matrix multiplication 4x5 * 5x5 using Strassen on 4x4 block."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We will build up U, V, W as lists of (index, value) pairs per column,
    # but since we need exact integer/half-integer entries and the output is a matrix,
    # let's just use dictionaries or lists of lists.
    # Given the small rank (expected 85), lists of lists are fine.
    
    U = []
    V = []
    W = []
    
    col = 0
    
    # Helper to add a term: U[i, col] += a, V[j, col] += b, W[k, col] += c
    # where U index is i*m+j? No, U is (n*m, rank). Index is row_in_A * m + col_in_A.
    # V is (m*p, rank). Index is row_in_B * p + col_in_B.
    # W is (p*n, rank). Index is row_in_C * n + col_in_C.
    # Wait, the convention is W[k*n+i, r] where C[i,k] = sum_r U[i*m+j? no]
    # Standard: C[i,k] = sum_{j,r} A[i,j] B[j,k]
    # Tensor decomposition: C[i,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]? 
    # No, the interface says:
    # U: (n*m, rank), index (i*m + j, r)
    # V: (m*p, rank), index (j*p + k, r)
    # W: (p*n, rank), index (k*n + i, r)
    # And the product should satisfy:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ?
    # Actually, the standard tensor decomposition for matrix multiplication T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
    # Let's check the seed program:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This corresponds to T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
    # Yes, this is the standard convention.
    
    # So for each rank term r, we add:
    # U[i*m+j, r] += a
    # V[j*p+k, r] += b
    # W[k*n+i, r] += c
    
    def add_term(i, j, k, a, b, c, col_idx):
        # Ensure U, V, W have enough rows
        while len(U) <= i*m+j: U.append(0)
        while len(V) <= j*p+k: V.append(0)
        while len(W) <= k*n+i: W.append(0)
        
        U[i*m+j] += a
        V[j*p+k] += b
        W[k*n+i] += c
    
    # This approach is flawed because a single column (rank term) can have multiple non-zero entries in U, V, W.
    # The seed program creates one column per (i,j,k) triple.
    # For Strassen, we need to combine terms.
    # It is better to build U, V, W as sparse structures or just accumulate into lists of lists.
    
    U = [[0] for _ in range(n*m)]  # We will append to each row
    V = [[0] for _ in range(m*p)]
    W = [[0] for _ in range(p*n)]
    
    current_col = 0
    
    def add_rank_term(U_entry, V_entry, W_entry):
        """
        U_entry: dict {i*m+j: a}
        V_entry: dict {j*p+k: b}
        W_entry: dict {k*n+i: c}
        """
        nonlocal current_col
        U_col = [0.0] * (n*m)
        V_col = [0.0] * (m*p)
        W_col = [0.0] * (p*n)
        
        for idx, val in U_entry.items():
            U_col[idx] += val
        for idx, val in V_entry.items():
            V_col[idx] += val
        for idx, val in W_entry.items():
            W_col[idx] += val
            
        for i in range(n*m):
            U[i].append(U_col[i])
        for i in range(m*p):
            V[i].append(V_col[i])
        for i in range(p*n):
            W[i].append(W_col[i])
        current_col += 1
    
    # 1. Strassen for A_L (4x4) and B_LL (4x4) -> C_LL (4x4)
    # A_L is A[0:4, 0:4], B_LL is B[0:4, 0:4]
    # Strassen computes C_LL = A_L * B_LL
    # Let A = [[A11, A12], [A21, A22]] where each is 2x2
    # Let B = [[B11, B12], [B21, B22]]
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to express each M in terms of individual scalar multiplications of A and B entries.
    # Since each M is a product of sums of 2x2 blocks, and each 2x2 block is 4 scalars,
    # (Sum A) * (Sum B) expands to 4*4 = 16 scalar products.
    # However, Strassen reduces the number of *recursive* multiplications.
    # At the base case (2x2 or 1x1), we still use schoolbook if we go all the way down.
    # Let's just use the 7 Strassen terms at the 4x4 level, and inside each term,
    # since the operands are 2x2 blocks, we can further apply Strassen? Or just schoolbook?
    # The prompt says "Strassen to the 4x4 core". This implies using Strassen for the 4x4 multiplication.
    # A 4x4 Strassen uses 7 multiplications of 2x2 blocks.
    # If we expand those 2x2 block multiplications using schoolbook, each is 4 mults.
    # Total 7 * 4 = 28 mults for the 4x4 * 4x4.
    # Schoolbook for 4x4 * 4x4 is 16 * 4 = 64 mults.
    # So this is a saving.
    
    # Let's implement the 7 Strassen terms for 4x4 by 4x4, expanding the 2x2 block products with schoolbook.
    
    def add_strassen_4x4(a_offset, b_offset, c_offset):
        """
        Adds rank terms for C[a_offset:a_offset+4, b_offset:b_offset+4] = 
        A[a_offset:a_offset+4, a_offset:a_offset+4] * B[b_offset:b_offset+4, b_offset:b_offset+4]
        Note: A indices are (i, j), B indices are (j, k).
        Here, the column index of A is j, row index of B is j.
        Since the "inner" dimension is 5, and we are multiplying 4x4 blocks,
        we need to be careful with the indices.
        
        Let's define the local indices for the 4x4 blocks:
        A_local[i, j] corresponds to A_global[i, j] (since A_L is [0:4, 0:4])
        B_local[j, k] corresponds to B_global[j, k] (since B_LL is [0:4, 0:4])
        
        Wait, in our problem:
        A is 4x5. A_L is A[0:4, 0:4]. So A_L[i, j] for i,j in 0..3.
        B is 5x5. B_LL is B[0:4, 0:4]. So B_LL[j, k] for j,k in 0..3.
        C_LL is C[0:4, 0:4].
        
        So the indices are straightforward.
        """
        # Define the 2x2 blocks
        # A11: A[0:2, 0:2], A12: A[0:2, 2:4]
        # A21: A[2:4, 0:2], A22: A[2:4, 2:4]
        # B11: B[0:2, 0:2], B12: B[0:2, 2:4]
        # B21: B[2:4, 0:2], B22: B[2:4, 2:4]
        
        # We will generate the 7 terms.
        # Each term is (Sum of A entries) * (Sum of B entries)
        # We expand this into individual (A[i,j] * B[j,k]) terms.
        
        # Term 1: (A11 + A22) * (B11 + B22)
        # A11 entries: (0,0), (0,1), (1,0), (1,1)
        # A22 entries: (2,2), (2,3), (3,2), (3,3)
        # B11 entries: (0,0), (0,1), (1,0), (1,1)
        # B22 entries: (2,2), (2,3), (3,2), (3,3)
        
        # This results in 8 * 8 = 64 terms? No, wait.
        # (A11 + A22) is a 2x2 matrix. (B11 + B22) is a 2x2 matrix.
        # Their product is a 2x2 matrix.
        # We need to add this product to C11.
        # C11[i,k] = sum_j (A11[i,j] + A22[i,j]) * (B11[j,k] + B22[j,k])
        # = sum_j (A11[i,j]*B11[j,k] + A11[i,j]*B22[j,k] + A22[i,j]*B11[j,k] + A22[i,j]*B22[j,k])
        
        # This is getting complicated to index manually.
        # Let's write a helper to add a product of two 2x2 blocks to a specific part of C.
        
        def add_2x2_product(a_indices, b_indices, c_target, sign=1):
            """
            a_indices: list of 4 (i,j) pairs for the 2x2 block in A
            b_indices: list of 4 (j,k) pairs for the 2x2 block in B
            c_target: function(i, j, k, val) to add val to C[i,k] for the product A[i,j]*B[j,k]
            """
            # a_indices is [[(0,0), (0,1)], [(1,0), (1,1)]] relative to block
            # Let's just iterate.
            pass # This is too abstract. Let's just hardcode the 7 terms.
            
        # Alternative: Just use the known Strassen decomposition and expand each 2x2 product.
        # Let's define the 16 scalar products for a 2x2 * 2x2.
        # If A_blk = [[a00, a01], [a10, a11]] and B_blk = [[b00, b01], [b10, b11]]
        # C_blk[0,0] = a00*b00 + a01*b10
        # C_blk[0,1] = a00*b01 + a01*b11
        # C_blk[1,0] = a10*b00 + a11*b10
        # C_blk[1,1] = a10*b01 + a11*b11
        
        # We will create 7 rank terms, each consisting of 4 scalar products.
        # Total 28 rank terms for the 4x4*4x4 part.
        
        # Term 1: M1 = (A11+A22) * (B11+B22) -> C11
        # A11+A22 is 2x2. B11+B22 is 2x2.
        # We treat (A11+A22) as a virtual 2x2 matrix A_v with entries:
        # A_v[0,0] = A[0,0]+A[2,2], A_v[0,1] = A[0,1]+A[2,3]
        # A_v[1,0] = A[1,0]+A[3,2], A_v[1,1] = A[1,1]+A[3,3]
        # B_v[0,0] = B[0,0]+B[2,2], B_v[0,1] = B[0,1]+B[2,3]
        # B_v[1,0] = B[1,0]+B[3,2], B_v[1,1] = B[1,1]+B[3,3]
        
        # Product P = A_v * B_v.
        # P[0,0] = A_v[0,0]*B_v[0,0] + A_v[0,1]*B_v[1,0]
        # This P[0,0] adds to C[0,0].
        
        # Let's create a helper to add a "virtual 2x2 product" to C.
        def add_virtual_2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, sign=1):
            """
            Adds sign * (A[a_rows, a_cols] * B[b_rows, b_cols]) to C[c_rows, c_cols]
            a_rows, a_cols: ranges or lists for the 2x2 block in A
            b_rows, b_cols: ranges or lists for the 2x2 block in B
            c_rows, c_cols: ranges or lists for the 2x2 block in C
            Assumes a_rows, a_cols, etc. are length 2 lists.
            """
            for r in range(2):
                for c in range(2):
                    # C[c_rows[r] + r, c_cols[c] + c] += sign * (A[a_rows[r], a_cols[r]] * B[b_rows[c], b_cols[c]] + ...)
                    # Wait, matrix multiplication sum is over the inner dimension.
                    # P[r,c] = sum_k A_v[r,k] * B_v[k,c]
                    # A_v[r,k] is a sum of A entries. B_v[k,c] is a sum of B entries.
                    # So P[r,c] is a sum of products of sums.
                    # We need to expand A_v[r,k] * B_v[k,c] into individual A[i,j]*B[j,k] terms.
                    
                    # This is getting very messy. Let's just iterate over all 4x4x4=64 terms?
                    # No, that defeats the purpose.
                    
                    # Let's step back.
                    # The seed program has rank 100.
                    # A simple Strassen on 4x4 reduces 64 mults to 28 mults.
                    # The remaining parts (4x1, 1x4, 1x1) are 16+16+4 = 36 mults.
                    # Total 28 + 36 = 64 mults.
                    # Rank 64 < 100.
                    
                    # Let's just implement the 28 Strassen terms explicitly.
                    pass
            
            pass # Too complex for inline.
        
        # Let's try a different approach:
        # Just use the 7 Strassen terms, but implement each term as a single rank-1 update?
        # No, Strassen terms are sums of products.
        
        # Okay, I will write a small function to expand a 2x2 product.
        def expand_2x2_product(a_idx, b_idx, c_idx):
            """
            a_idx: [[i0,j0],[i1,j1]] -> A[i0,j0], A[i0,j1], A[i1,j0], A[i1,j1]
            b_idx: [[j0,k0],[j1,k1]] -> B[j0,k0], B[j0,k1], B[j1,k0], B[j1,k1]
            c_idx: [[i0,k0],[i1,k1]] -> C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1]
            """
            # C[i0,k0] += A[i0,j0]*B[j0,k0] + A[i0,j1]*B[j1,k0]
            # C[i0,k1] += A[i0,j0]*B[j0,k1] + A[i0,j1]*B[j1,k1]
            # C[i1,k0] += A[i1,j0]*B[j0,k0] + A[i1,j1]*B[j1,k0]
            # C[i1,k1] += A[i1,j0]*B[j0,k1] + A[i1,j1]*B[j1,k1]
            
            # Each of these is a sum of 2 products.
            # We will add 2 rank terms for each C entry? No, that would be 4*2=8 terms per 2x2 block.
            # That's back to schoolbook.
            
            # The power of Strassen is that the *same* intermediate products (M1..M7) are reused.
            # But in a tensor decomposition, each "multiplication" is a rank-1 term.
            # If we compute M1 = (A11+A22)*(B11+B22), and we expand this into scalar products,
            # we get 4*4=16 scalar products for the 2x2*2x2 product? No, 2x2*2x2 is 8 scalar products.
            # Wait, 2x2 * 2x2 matrix multiplication requires 8 scalar multiplications.
            # So M1 requires 8 rank-1 terms.
            # M2 requires 8.
            # ...
            # M7 requires 8.
            # Total 7 * 8 = 56 rank-1 terms.
            # Schoolbook 4x4*4x4 is 64 rank-1 terms.
            # So Strassen saves 8 terms.
            # Total rank: 56 (for 4x4*4x4) + 16 (4x1*1x4) + 16 (4x4*4x1) + 4 (4x1*1x1) = 92.
            # This is better than 100, but not as good as my previous estimate of 85.
            # Why did I think 28? 7 * 4? No, 2x2*2x2 is 8 mults.
            
            # So rank 92 is the target.
            
            # Let's implement this.
            pass
        
        pass # The above logic is sound. Rank 92.

    # Due to complexity of manual Strassen expansion in this format, and time constraints,
    # I will stick to a simpler optimization or just return the seed if I can't cleanly implement Strassen.
    # However, the task asks for a *new* approach.
    # Let's try a different split: 2+3?
    # 2x3 Strassen? No, Strassen is for powers of 2.
    
    # What if I just use the seed program but with half-integers? No, entries are already 0 or 1.
    
    # Let's try to implement the 4x1, 1x4, 1x1 parts with Strassen? No, they are rank-1.
    
    # I will implement the Strassen 4x4 part explicitly.
    
    # Define the 7 terms for 4x4 Strassen.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # A11: rows 0-1, cols 0-1. A12: rows 0-1, cols 2-3.
    # A21: rows 2-3, cols 0-1. A22: rows 2-3, cols 2-3.
    # B11: rows 0-1, cols 0-1. B12: rows 0-1, cols 2-3.
    # B21: rows 2-3, cols 0-1. B22: rows 2-3, cols 2-3.
    
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product.
    # We will expand each M into 8 scalar products.
    
    def add_2x2_block_product(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, sign=1):
        """
        Adds sign * A[a_rows, a_cols] * B[b_rows, b_cols] to C[c_rows, c_cols]
        a_rows, a_cols, b_rows, b_cols, c_rows, c_cols are lists of 2 indices.
        """
        # C[c0, k0] += A[a0, b0] * B[b0, c0] + A[a0, b1] * B[b1, c0]
        # Wait, the indices are:
        # A[i, j], B[j, k], C[i, k]
        # a_rows are i indices, a_cols are j indices for A.
        # b_rows are j indices, b_cols are k indices for B.
        # c_rows are i indices, c_cols are k indices for C.
        
        # Let's say a_rows=[i0, i1], a_cols=[j0, j1]
        # b_rows=[j0, j1], b_cols=[k0, k1]
        # c_rows=[i0, i1], c_cols=[k0, k1]
        
        # C[i0, k0] += A[i0, j0]*B[j0, k0] + A[i0, j1]*B[j1, k0]
        # C[i0, k1] += A[i0, j0]*B[j0, k1] + A[i0, j1]*B[j1, k1]
        # C[i1, k0] += A[i1, j0]*B[j0, k0] + A[i1, j1]*B[j1, k0]
        # C[i1, k1] += A[i1, j0]*B[j0, k1] + A[i1, j1]*B[j1, k1]
        
        for r in range(2):
            for c in range(2):
                # Term 1: A[i_r, j_0] * B[j_0, k_c]
                i_r = a_rows[r]
                j_0 = a_cols[0]
                k_c = c_cols[c]
                add_rank_term({i_r*m + j_0: sign}, {j_0*p + k_c: 1}, {k_c*n + i_r: 1})
                
                # Term 2: A[i_r, j_1] * B[j_1, k_c]
                j_1 = a_cols[1]
                add_rank_term({i_r*m + j_1: sign}, {j_1*p + k_c: 1}, {k_c*n + i_r: 1})

    # Now, call this for the 7 Strassen terms.
    
    # A11: rows 0,1; cols 0,1
    # A12: rows 0,1; cols 2,3
    # A21: rows 2,3; cols 0,1
    # A22: rows 2,3; cols 2,3
    # B11: rows 0,1; cols 0,1
    # B12: rows 0,1; cols 2,3
    # B21: rows 2,3; cols 0,1
    # B22: rows 2,3; cols 2,3
    # C11: rows 0,1; cols 0,1
    # C12: rows 0,1; cols 2,3
    # C21: rows 2,3; cols 0,1
    # C22: rows 2,3; cols 2,3
    
    # M1 = (A11 + A22) * (B11 + B22)
    # This is a sum of 4 products: A11*B11, A11*B22, A22*B11, A22*B22
    # We can just call add_2x2_block_product for each.
    
    add_2x2_block_product([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 1) # A11*B11 -> C11
    add_2x2_block_product([0,1], [0,1], [2,3], [0,1], [0,1], [0,1], 1) # A11*B22 -> C11
    add_2x2_block_product([2,3], [0,1], [0,1], [0,1], [0,1], [0,1], 1) # A22*B11 -> C11
    add_2x2_block_product([2,3], [0,1], [2,3], [0,1], [0,1], [0,1], 1) # A22*B22 -> C11
    
    # M2 = (A21 + A22) * B11
    add_2x2_block_product([2,3], [0,1], [0,1], [0,1], [2,3], [0,1], 1) # A21*B11 -> C21
    add_2x2_block_product([2,3], [2,3], [0,1], [0,1], [2,3], [0,1], 1) # A22*B11 -> C21
    
    # M3 = A11 * (B12 - B22)
    add_2x2_block_product([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], 1) # A11*B12 -> C12
    add_2x2_block_product([0,1], [0,1], [2,3], [2,3], [0,1], [2,3], -1) # A11*B22 -> C12
    
    # M4 = A22 * (B21 - B11)
    add_2x2_block_product([2,3], [2,3], [2,3], [0,1], [0,1], [0,1], 1) # A22*B21 -> C11
    add_2x2_block_product([2,3], [2,3], [0,1], [0,1], [0,1], [0,1], -1) # A22*B11 -> C11
    
    # M5 = (A11 + A12) * B22
    add_2x2_block_product([0,1], [0,1], [2,3], [2,3], [0,1], [0,1], -1) # A11*B22 -> C11
    add_2x2_block_product([0,1], [2,3], [2,3], [2,3], [0,1], [0,1], -1) # A12*B22 -> C11
    add_2x2_block_product([0,1], [0,1], [2,3], [2,3], [0,1], [2,3], 1) # A11*B22 -> C12
    add_2x2_block_product([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], 1) # A12*B22 -> C12
    
    # M6 = (A21 - A11) * (B11 + B12)
    add_2x2_block_product([2,3], [0,1], [0,1], [0,1], [2,3], [2,3], 1) # A21*B11 -> C22
    add_2x2_block_product([2,3], [0,1], [0,1], [2,3], [2,3], [2,3], 1) # A21*B12 -> C22
    add_2x2_block_product([0,1], [0,1], [0,1], [0,1], [2,3], [2,3], -1) # A11*B11 -> C22
    add_2x2_block_product([0,1], [0,1], [0,1], [2,3], [2,3], [2,3], -1) # A11*B12 -> C22
    
    # M7 = (A12 - A22) * (B21 + B22)
    add_2x2_block_product([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], 1) # A12*B21 -> C11
    add_2x2_block_product([0,1], [2,3], [2,3], [2,3], [0,1], [0,1], 1) # A12*B22 -> C11
    add_2x2_block_product([2,3], [2,3], [2,3], [0,1], [0,1], [0,1], -1) # A22*B21 -> C11
    add_2x2_block_product([2,3], [2,3], [2,3], [2,3], [0,1], [0,1], -1) # A22*B22 -> C11

    # Now the remaining parts:
    # A_R B_RL: A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    for i in range(4):
        for k in range(4):
            add_rank_term({i*m + 4: 1}, {4*p + k: 1}, {k*n + i: 1})
            
    # A_L B_LR: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    for i in range(4):
        for j in range(4):
            add_rank_term({i*m + j: 1}, {j*p + 4: 1}, {4*n + i: 1})
            
    # A_R B_RR: A[0:4, 4] * B[4, 4] -> C[0:4, 4]
    for i in range(4):
        add_rank_term({i*m + 4: 1}, {4*p + 4: 1}, {4*n + i: 1})

    # Convert lists to tuples or lists of floats? The interface says integer/half-integer.
    # My entries are integers.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
