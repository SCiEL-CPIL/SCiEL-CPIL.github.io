"""Block-based matrix multiplication decomposition using Strassen for 2x2x2 block."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    
    # Block structure:
    # A is 3x3: rows 0,1 are "top" (i=0,1), row 2 is "bottom" (i=2)
    #              cols 0,1 are "left" (j=0,1), col 2 is "right" (j=2)
    # B is 3x6:  rows 0,1 are "top" (j=0,1), row 2 is "bottom" (j=2)
    #              cols 0..5 are the 6 output columns (k=0..5)
    
    # C is 3x6: rows 0,1 are "top" (i=0,1), row 2 is "bottom" (i=2)
    #              cols 0..5 (k=0..5)
    
    # Blocks:
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2]  -> 2x2 x 2x2 -> Strassen rank 7
    # C[0:2, 2:6] = A[0:2, 2:2] * B[2:2, 2:6]  -> 2x1 x 1x4 -> rank 8
    # C[2:3, 0:2] = A[2:3, 0:2] * B[0:2, 0:2]  -> 1x2 x 2x2 -> rank 4
    # C[2:3, 2:6] = A[2:3, 2:3] * B[2:3, 2:6]  -> 1x1 x 1x4 -> rank 4
    #
    # Wait, let me re-examine. B is 3x6, so B[j, k] for j in 0,1,2 and k in 0..5.
    # The middle dimension of B is 3 (rows), which corresponds to m=3.
    # 
    # A[i,j] * B[j,k] = C[i,k]
    #
    # Let me re-partition more carefully:
    # A[0:2, 0:2] (2x2) * B[0:2, 0:6] (2x6) -> contributes to C[0:2, 0:6]
    # A[0:2, 2:3] (2x1) * B[2:3, 0:6] (1x6) -> contributes to C[0:2, 0:6]
    # A[2:3, 0:2] (1x2) * B[0:2, 0:6] (2x6) -> contributes to C[2:3, 0:6]
    # A[2:3, 2:3] (1x1) * B[2:3, 0:6] (1x6) -> contributes to C[2:3, 0:6]
    #
    # So:
    # C[0:2, :] = A[0:2, 0:2] * B[0:2, :] + A[0:2, 2] * B[2, :]
    # C[2:3, :] = A[2, 0:2] * B[0:2, :] + A[2, 2] * B[2, :]
    #
    # Block 1: A11(2x2) * B1(2x6) -> 2x6 output. This is 2*(2*6)=24 schoolbook, but we can use Strassen on the 2x2 part with each "column" of B being a 2-vector.
    # Actually, for 2x2 * 2x6, we can treat it as 3 independent 2x2*2x1 multiplications (one per group of... no, 6 columns).
    # 
    # Better approach: Think of it as multiplying a 2x2 matrix by a 2x6 matrix.
    # For each column of the result (6 columns), we need to compute a 2x1 = 2x2 * 2x1 product.
    # A 2x2 * 2x1 product has rank... let's think.
    # c1 = a11*b1 + a12*b2, c2 = a21*b1 + a22*b2
    # This is rank 4 schoolbook. Can we do better?
    # The tensor is a11*b1*c1 + a12*b2*c1 + a21*b1*c2 + a22*b2*c2
    # This is 4 independent terms, rank 4.
    #
    # So for 2x2 * 2x6: 6 * 4 = 24 terms schoolbook.
    # 
    # Hmm, that's not better. Let me reconsider.
    #
    # Alternative: Use Strassen's 7 products for the 2x2*2x2 core, and handle the remaining columns.
    #
    # Actually, let me just carefully compute the total rank by blocks:
    #
    # The multiplication C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # i in {0,1,2}, j in {0,1,2}, k in {0,...,5}
    #
    # Total terms: 3*3*6 = 54 (schoolbook)
    #
    # Let me group by (i, j) pairs and see which B[j,k] values they use:
    # For fixed (i,j), the term is A[i,j] * B[j,k] * C[i,k] for k=0..5
    # This is an outer product of the vector [A[i,j]] (scalar) with [B[j,0],...,B[j,5]] (6-vector) producing [C[i,0],...,C[i,5]] (6-vector contribution)
    # Each such (i,j) pair contributes rank 6 (since A[i,j] is a single variable, and B[j,k] for k=0..5 are 6 variables).
    #
    # So we have 9 (i,j) pairs, each contributing rank 6, total 54.
    #
    # To improve, we need to combine terms. The key insight is that for a fixed j, we have multiple i values sharing the same B[j,k] variables.
    #
    # For j=0: i=0,1,2 all use B[0,k]
    #   C[i,k] += A[i,0] * B[0,k] for i=0,1,2, k=0..5
    #   This is a rank 18 block: 3 A-variables times 6 B-variables
    #   Can we do better? The tensor is sum_{i,k} A[i,0]*B[0,k]*C[i,k]
    #   This is rank 18 (all 18 monomials are independent).
    #
    # Similarly for j=1: rank 18
    # For j=2: rank 18
    # Total: 54.
    #
    # So grouping by j doesn't help directly. We need to find linear combinations.
    #
    # Let me try a different block decomposition:
    # 
    # Partition i into {0,1} and {2}, j into {0,1} and {2}:
    #
    # C[0:2, :] = A[0:2, 0:2] * B[0:2, :] + A[0:2, 2] * B[2, :]
    # C[2, :] = A[2, 0:2] * B[0:2, :] + A[2, 2] * B[2, :]
    #
    # Block A: A[0:2,0:2] (2x2) * B[0:2,:] (2x6)
    # This is a 2x2 matrix times a 2x6 matrix.
    # For each column k of B, we have a 2x1 vector, and we compute 2x2 * 2x1 = 2x1.
    # The 2x2 * 2x1 multiplication has rank 4 (as shown above).
    # So 6 columns * rank 4 = 24.
    #
    # But wait, can we use Strassen-like tricks? The 2x2 * 2x1 multiplication:
    # c1 = a11*b1 + a12*b2
    # c2 = a21*b1 + a22*b2
    # This is rank 4, and I don't see a way to reduce it below 4 since all 4 monomials are independent.
    #
    # Block B: A[0:2,2] (2x1) * B[2,:] (1x6)
    # C[i,k] += A[i,2] * B[2,k] for i=0,1, k=0..5
    # This is rank 12 (2 A-vars * 6 B-vars).
    #
    # Block C: A[2,0:2] (1x2) * B[0:2,:] (2x6)
    # C[2,k] += A[2,0]*B[0,k] + A[2,1]*B[1,k] for k=0..5
    # This is rank 12 (2 A-vars * 6 B-vars).
    #
    # Block D: A[2,2] (1x1) * B[2,:] (1x6)
    # C[2,k] += A[2,2]*B[2,k] for k=0..5
    # This is rank 6.
    #
    # Total: 24 + 12 + 12 + 6 = 54. Same as schoolbook.
    #
    # The issue is that the 2x2*2x6 block has rank 24, same as schoolbook for that subproblem.
    #
    # Let me try yet another approach. What if I use a different partition?
    #
    # What if I partition k (columns of B, rows of C) into groups?
    # 
    # Actually, the key to Strassen is that for 2x2*2x2, you can do it in 7 products instead of 8.
    # The 2x2*2x2 case has 8 terms in schoolbook, and Strassen reduces to 7.
    #
    # For 2x2*2x6, the schoolbook is 24. Can we reduce this?
    # The 2x2*2x6 can be seen as 3 independent 2x2*2x2 multiplications if we partition the 6 columns into 3 groups of 2.
    # Each 2x2*2x2 can be done in 7, so 3*7 = 21.
    # 
    # Yes! That's the key. Let me partition the 6 output columns into 3 groups of 2.
    #
    # For each group of 2 columns, we have a 2x2*2x2 multiplication (where the "2x2" on the B side is actually 2 rows x 2 columns).
    #
    # Let me redefine:
    # B[0:2, 0:2] is 2x2, B[0:2, 2:4] is 2x2, B[0:2, 4:6] is 2x2
    #
    # C[0:2, 0:2] = A[0:2,0:2] * B[0:2,0:2] -> Strassen rank 7
    # C[0:2, 2:4] = A[0:2,0:2] * B[0:2,2:4] -> Strassen rank 7
    # C[0:2, 4:6] = A[0:2,0:2] * B[0:2,4:6] -> Strassen rank 7
    #
    # Total for A[0:2,0:2] * B[0:2,:]: 21
    #
    # C[0:2, :] also gets contribution from A[0:2,2] * B[2,:]
    # A[0:2,2] is 2x1, B[2,:] is 1x6
    # C[i,k] += A[i,2]*B[2,k] for i=0,1, k=0..5
    # Rank 12
    #
    # C[2, :] = A[2,0:2] * B[0:2,:] + A[2,2] * B[2,:]
    # A[2,0:2] is 1x2, B[0:2,:] is 2x6
    # C[2,k] += A[2,0]*B[0,k] + A[2,1]*B[1,k] for k=0..5
    # Rank 12
    #
    # A[2,2] * B[2,:] is 1x1 * 1x6, rank 6
    #
    # Total: 21 + 12 + 12 + 6 = 51
    #
    # That's an improvement from 54 to 51!
    #
    # Can we do better? Let's see if we can apply Strassen to other blocks.
    #
    # The 1x2 * 2x6 block (A[2,0:2] * B[0:2,:]):
    # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] for k=0..5
    # This is rank 12. Can we reduce it?
    # The tensor is sum_{k} (A[2,0]*B[0,k] + A[2,1]*B[1,k]) * C[2,k]
    # = sum_{k} A[2,0]*B[0,k]*C[2,k] + sum_{k} A[2,1]*B[1,k]*C[2,k]
    # These 12 monomials are all independent, so rank is 12.
    #
    # The 2x1 * 1x6 block (A[0:2,2] * B[2,:]):
    # C[i,k] = A[i,2]*B[2,k] for i=0,1, k=0..5
    # Rank 12, all independent.
    #
    # The 1x1 * 1x6 block: rank 6, all independent.
    #
    # So 51 seems to be the best with this approach.
    #
    # Let me implement the rank 51 decomposition.

    def strassen_2x2_x_2x2(a_row0, a_row1, b_col0, b_col1, b_col2, b_col3, W_base, U_base, V_base, k_start):
        """
        Compute 2x2 * 2x2 using Strassen's 7 products.
        
        A = [[a_row0[0], a_row0[1]], [a_row1[0], a_row1[1]]]
        B = [[b_col0[0], b_col1[0]], [b_col2[0], b_col3[0]]]  # b_colX are the k-indices for a specific k-group
        Wait, I need to be more careful.
        
        Let me define it differently. For a specific group of 2 output columns (k_start, k_start+1):
        B_sub = B[0:2, k_start:k_start+2] which is a 2x2 matrix.
        B_sub[0,0] = B[0, k_start], B_sub[0,1] = B[0, k_start+1]
        B_sub[1,0] = B[1, k_start], B_sub[1,1] = B[1, k_start+1]
        
        A_sub = A[0:2, 0:2] which is a 2x2 matrix.
        A_sub[0,0] = A[0,0], A_sub[0,1] = A[0,1]
        A_sub[1,0] = A[1,0], A_sub[1,1] = A[1,1]
        
        Strassen products:
        M1 = (A[0,0]+A[1,1]) * (B[0,k]+B[1,k+1])  where k=k_start
        M2 = (A[1,0]+A[1,1]) * B[0,k]
        M3 = A[0,0] * (B[0,k+1]-B[1,k+1])
        M4 = A[1,1] * (B[1,k]-B[0,k])
        M5 = (A[0,0]+A[0,1]) * B[1,k+1]
        M6 = (A[1,0]-A[0,0]) * (B[0,k]+B[0,k+1])
        M7 = (A[0,1]-A[1,1]) * (B[1,k]+B[1,k+1])
        
        Results:
        C[0,k]   = M1 + M4 - M5 + M7
        C[0,k+1] = M3 + M5
        C[1,k]   = M2 + M4
        C[1,k+1] = M1 - M2 + M3 + M6
        """
        # We'll add 7 columns to U, V, W
        cols = []
        for _ in range(7):
            cols.append(k_start)
            k_start += 1
        
        # For each Strassen product, define the U, V, W entries
        # U is indexed by (i*m + j) where i is row of A, j is col of A
        # V is indexed by (j*p + k) where j is row of B, k is col of B
        # W is indexed by (k*n + i) where k is col of B (which is col of C), i is row of C
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k]+B[1,k+1])
        # U: A[0,0] -> U[0*3+0] = U[0], A[1,1] -> U[1*3+1] = U[4]
        # V: B[0,k] -> V[0*6+k], B[1,k+1] -> V[1*6+k+1]
        # W: C[0,k], C[0,k+1], C[1,k], C[1,k+1]
        #   C[0,k] -> W[k*3+0] = W[3k]
        #   C[0,k+1] -> W[(k+1)*3+0] = W[3k+3]
        #   C[1,k] -> W[k*3+1] = W[3k+1]
        #   C[1,k+1] -> W[(k+1)*3+1] = W[3k+4]
        
        k1 = cols[0]
        # M1: (A00+A11)*(B0k+B1k1)
        U[k1][0] += 1   # A[0,0]
        U[k1][4] += 1   # A[1,1]
        V[k1][0*6 + (k1 % 6 if k1 < 6 else 0)]  # This is getting complicated. Let me use a different approach.
        
        return k_start
    
    # This is getting too complex with inline Strassen. Let me build it more carefully.
    
    # I'll construct the full U, V, W matrices by adding contributions from each block.
    
    total_rank = 0
    U_rows = n * m  # 9
    V_rows = m * p  # 18
    W_rows = p * n  # 18
    
    # Start with empty matrices
    U = [[0] * 1 for _ in range(U_rows)]  # We'll grow this
    V = [[0] * 1 for _ in range(V_rows)]
    W = [[0] * 1 for _ in range(W_rows)]
    
    def add_column(U_val, V_val, W_val):
        nonlocal total_rank
        idx = total_rank
        for i in range(U_rows):
            U[i].append(0)
        for i in range(V_rows):
            V[i].append(0)
        for i in range(W_rows):
            W[i].append(0)
        for i, v in U_val.items():
            U[i][idx] += v
        for i, v in V_val.items():
            V[i][idx] += v
        for i, v in W_val.items():
            W[i][idx] += v
        total_rank += 1
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (Strassen, rank 7)
    # k_start = 0 for first column group
    k0, k1 = 0, 1
    # M1 = (A00+A11)*(B0k0+B1k1)
    add_column(
        {0: 1, 4: 1},  # U: A[0,0]=idx 0, A[1,1]=idx 4
        {0*6+k0: 1, 1*6+k1: 1},  # V: B[0,0], B[1,1]
        {k0*3+0: 1, k1*3+0: 1, k0*3+1: 1, k1*3+1: 1}  # W: C[0,k0], C[0,k1], C[1,k0], C[1,k1]
    )
    # M2 = (A10+A11)*B0k0
    add_column(
        {3: 1, 4: 1},  # U: A[1,0]=idx 3, A[1,1]=idx 4
        {0*6+k0: 1},  # V: B[0,0]
        {k0*3+1: 1, k1*3+1: -1}  # W: C[1,k0], C[1,k1] (M2 contributes + to C[1,k0], - to C[1,k1])
    )
    # M3 = A00*(B0k1-B1k1)
    add_column(
        {0: 1},  # U: A[0,0]
        {0*6+k1: 1, 1*6+k1: -1},  # V: B[0,1], B[1,1]
        {k1*3+0: 1, k1*3+1: 1}  # W: C[0,k1], C[1,k1] (M3 contributes + to both)
    )
    # M4 = A11*(B1k0-B0k0)
    add_column(
        {4: 1},  # U: A[1,1]
        {1*6+k0: 1, 0*6+k0: -1},  # V: B[1,0], B[0,0]
        {k0*3+0: 1, k0*3+1: 1}  # W: C[0,k0], C[1,k0] (M4 contributes + to both)
    )
    # M5 = (A00+A01)*B1k1
    add_column(
        {0: 1, 1: 1},  # U: A[0,0], A[0,1]
        {1*6+k1: 1},  # V: B[1,1]
        {k0*3+0: -1, k1*3+0: 1}  # W: C[0,k0] gets -M5, C[0,k1] gets +M5
    )
    # M6 = (A10-A00)*(B0k0+B0k1)
    add_column(
        {3: 1, 0: -1},  # U: A[1,0], A[0,0]
        {0*6+k0: 1, 0*6+k1: 1},  # V: B[0,0], B[0,1]
        {k1*3+1: 1}  # W: C[1,k1] gets +M6
    )
    # M7 = (A01-A11)*(B1k0+B1k1)
    add_column(
        {1: 1, 4: -1},  # U: A[0,1], A[1,1]
        {1*6+k0: 1, 1*6+k1: 1},  # V: B[1,0], B[1,1]
        {k0*3+0: 1}  # W: C[0,k0] gets +M7
    )
    
    # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] (Strassen, rank 7)
    k0, k1 = 2, 3
    add_column(
        {0: 1, 4: 1},
        {0*6+k0: 1, 1*6+k1: 1},
        {k0*3+0: 1, k1*3+0: 1, k0*3+1: 1, k1*3+1: 1}
    )
    add_column(
        {3: 1, 4: 1},
        {0*6+k0: 1},
        {k0*3+1: 1, k1*3+1: -1}
    )
    add_column(
        {0: 1},
        {0*6+k1: 1, 1*6+k1: -1},
        {k1*3+0: 1, k1*3+1: 1}
    )
    add_column(
        {4: 1},
        {1*6+k0: 1, 0*6+k0: -1},
        {k0*3+0: 1, k0*3+1: 1}
    )
    add_column(
        {0: 1, 1: 1},
        {1*6+k1: 1},
        {k0*3+0: -1, k1*3+0: 1}
    )
    add_column(
        {3: 1, 0: -1},
        {0*6+k0: 1, 0*6+k1: 1},
        {k1*3+1: 1}
    )
    add_column(
        {1: 1, 4: -1},
        {1*6+k0: 1, 1*6+k1: 1},
        {k0*3+0: 1}
    )
    
    # Block 3: A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6] (Strassen, rank 7)
    k0, k1 = 4, 5
    add_column(
        {0: 1, 4: 1},
        {0*6+k0: 1, 1*6+k1: 1},
        {k0*3+0: 1, k1*3+0: 1, k0*3+1: 1, k1*3+1: 1}
    )
    add_column(
        {3: 1, 4: 1},
        {0*6+k0: 1},
        {k0*3+1: 1, k1*3+1: -1}
    )
    add_column(
        {0: 1},
        {0*6+k1: 1, 1*6+k1: -1},
        {k1*3+0: 1, k1*3+1: 1}
    )
    add_column(
        {4: 1},
        {1*6+k0: 1, 0*6+k0: -1},
        {k0*3+0: 1, k0*3+1: 1}
    )
    add_column(
        {0: 1, 1: 1},
        {1*6+k1: 1},
        {k0*3+0: -1, k1*3+0: 1}
    )
    add_column(
        {3: 1, 0: -1},
        {0*6+k0: 1, 0*6+k1: 1},
        {k1*3+1: 1}
    )
    add_column(
        {1: 1, 4: -1},
        {1*6+k0: 1, 1*6+k1: 1},
        {k0*3+0: 1}
    )
    
    # Block 4: A[0:2,2] * B[2,:] -> C[0:2,:] (rank 12)
    # A[0,2] = U idx 2, A[1,2] = U idx 5
    # B[2,k] = V idx 2*6+k = 12+k for k=0..5
    # C[0,k] = W idx k*3+0, C[1,k] = W idx k*3+1
    for k in range(6):
        # C[0,k] += A[0,2]*B[2,k]
        add_column(
            {2: 1},
            {12+k: 1},
            {k*3+0: 1}
        )
        # C[1,k] += A[1,2]*B[2,k]
        add_column(
            {5: 1},
            {12+k: 1},
            {k*3+1: 1}
        )
    
    # Block 5: A[2,0:2] * B[0:2,:] -> C[2,:] (rank 12)
    # A[2,0] = U idx 6, A[2,1] = U idx 7
    # B[0,k] = V idx 0*6+k = k, B[1,k] = V idx 1*6+k = 6+k
    # C[2,k] = W idx k*3+2
    for k in range(6):
        # C[2,k] += A[2,0]*B[0,k]
        add_column(
            {6: 1},
            {k: 1},
            {k*3+2: 1}
        )
        # C[2,k] += A[2,1]*B[1,k]
        add_column(
            {7: 1},
            {6+k: 1},
            {k*3+2: 1}
        )
    
    # Block 6: A[2,2] * B[2,:] -> C[2,:] (rank 6)
    # A[2,2] = U idx 8
    # B[2,k] = V idx 12+k
    # C[2,k] = W idx k*3+2
    for k in range(6):
        add_column(
            {8: 1},
            {12+k: 1},
            {k*3+2: 1}
        )
    
    # Total rank should be 7+7+7+12+12+6 = 51
    # But we started with 1 dummy column, so we need to remove it.
    # Actually, let's just trim it.
    
    rank = total_rank
    U = [row[1:] for row in U]  # Remove dummy column
    V = [row[1:] for row in V]
    W = [row[1:] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

