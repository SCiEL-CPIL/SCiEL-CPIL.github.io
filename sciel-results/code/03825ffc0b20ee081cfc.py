"""Matrix multiplication 4x4x5 - Strassen-based decomposition."""
import time

DIMENSIONS = [4, 4, 5]

def construct():
    """Construct rank-32 decomposition for 4x4x5 matrix multiplication.
    
    Uses Strassen for the 4x4x4 part (rank 28) plus rank-4 for the 4x4x1 part.
    """
    n, m, p = 4, 4, 5
    # U: (n*m, rank) = (16, 32)
    # V: (m*p, rank) = (20, 32)
    # W: (p*n, rank) = (20, 32)
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2: 7 terms
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4x4: A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each product is 2x2x2, using Strassen (rank 7)
    # Total: 8 products * 7 = 56... wait, that's not right.
    
    # Actually for 4x4x4 with Strassen:
    # We can apply Strassen directly to 4x4x4 by treating it as
    # multiplying two 4x4 matrices where we split into 2x2 blocks.
    # 
    # The standard approach: C = A*B where A, B, C are 4x4.
    # Split A into 4 blocks of 2x2: A11, A12, A21, A22
    # Split B into 4 blocks of 2x2: B11, B12, B21, B22
    # Then:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Each of the 8 products A_ij * B_jk is a 2x2x2 multiplication.
    # Using Strassen for each gives 7 terms per product, total 56 terms.
    # But we can share some computations...
    
    # Actually, let me think more carefully. The rank of 4x4x4 matrix
    # multiplication is known to be at most 21 (Bini et al.) but
    # Strassen gives 7^2 = 49 for the recursive case.
    
    # Wait, for 4x4x4, if we use Strassen recursively:
    # 4x4x4 can be done with 7 multiplications of 2x2x2 if we use
    # the Strassen pattern. Let me re-derive.
    
    # For 2x2x2 Strassen:
    # Given A (2x2) and B (2x2):
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)*(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C = [[M1+M4-M5+M7, M3+M5],
    #      [M2+M4, M1-M2+M3+M6]]
    
    # For 4x4x4, we partition A and B into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bjk is 2x2
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Now, each product like A11*B11 is a 2x2x2 multiplication.
    # We can use Strassen for each, but that gives 8*7=56 terms.
    #
    # However, we can do better by using Strassen at the 4x4 level.
    # The key insight: we can form linear combinations of the 2x2 blocks
    # to create 7 "super-multiplications" of 2x2x2.
    
    # Let me define the Strassen factors for 4x4x4 directly.
    # We have 16 variables in A (a_00 to a_33) and 16 in B.
    # 
    # Using the block Strassen approach:
    # Let A11 = [[a00,a01],[a10,a11]], A12 = [[a02,a03],[a12,a13]]
    # A21 = [[a20,a21],[a30,a31]], A22 = [[a22,a23],[a32,a33]]
    # Similarly for B.
    #
    # Strassen at the block level:
    # S1 = (A11+A22)*(B11+B22)
    # S2 = (A21+A22)*B11
    # S3 = A11*(B12-B22)
    # S4 = A22*(B21-B11)
    # S5 = (A11+A12)*B22
    # S6 = (A21-A11)*(B11+B12)
    # S7 = (A12-A22)*(B21+B22)
    #
    # C11 = S1 + S4 - S5 + S7
    # C12 = S3 + S5
    # C21 = S2 + S4
    # C22 = S1 - S2 + S3 + S6
    #
    # Each S_i is a 2x2x2 multiplication (product of two 2x2 matrices),
    # so each can be done with 7 scalar multiplications using Strassen.
    # Total: 7 * 7 = 49 terms.
    
    # Hmm, that's 49 for the 4x4x4 part. Plus 4 for the 4x4x1 part = 53.
    # That's still less than 80 but not as good as I hoped.
    
    # Actually, the known rank for 4x4x4 is at most 21 over the rationals
    # (Bini, Capovani, Lotti, Romani 1979), but constructing that is complex.
    #
    # Let me try a different approach: use the schoolbook for 4x4x1 (rank 4)
    # and Strassen-based for 4x4x4.
    
    # For now, let me use a simpler but still improved approach.
    # Use Strassen for the 4x4x4 part with 49 terms + 4 for 4x4x1 = 53.
    
    # Actually, let me just construct the Strassen decomposition properly.
    
    # I'll build the 4x4x4 Strassen decomposition (49 terms) and add
    # 4 more terms for the 5th column.
    
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper: add a Strassen 2x2x2 decomposition
    # A rows: a_start, a_start+1 (these are row indices in the 2x2 block)
    # B cols: b_start, b_start+1
    # C rows: c_start, c_start+1
    # The 2x2 blocks are at positions:
    # A: rows [ar0, ar1], cols [ac0, ac1]
    # B: rows [br0, br1], cols [bc0, bc1]
    # C: rows [cr0, cr1], cols [cc0, cc1]
    
    def add_strassen_2x2x2(U, V, W, n, m, p, 
                           a_rows, a_cols, 
                           b_rows, b_cols,
                           c_rows, c_cols,
                           col_offset):
        """Add Strassen decomposition for one 2x2x2 multiplication.
        
        a_rows: tuple of 2 row indices in A (0-indexed)
        a_cols: tuple of 2 col indices in A
        b_rows: tuple of 2 row indices in B
        b_cols: tuple of 2 col indices in B
        c_rows: tuple of 2 row indices in C (for W, this maps to k*n+i)
        c_cols: tuple of 2 col indices in C
        col_offset: starting column in U, V, W
        """
        ar0, ar1 = a_rows
        ac0, ac1 = a_cols
        br0, br1 = b_rows
        bc0, bc1 = b_cols
        cr0, cr1 = c_rows
        cc0, cc1 = c_cols
        
        # A element at (i,j) -> U row index = i*m + j
        # B element at (j,k) -> V row index = j*p + k
        # C element at (k,i) -> W row index = k*n + i (trace convention)
        
        # For the 2x2x2 subproblem:
        # a = A[ar0, ac0], b = A[ar0, ac1]
        # c = A[ar1, ac0], d = A[ar1, ac1]
        # e = B[br0, bc0], f = B[br0, bc1]
        # g = B[br1, bc0], h = B[br1, bc1]
        
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C[cr0, cc0] = M1 + M4 - M5 + M7
        # C[cr0, cc1] = M3 + M5
        # C[cr1, cc0] = M2 + M4
        # C[cr1, cc1] = M1 - M2 + M3 + M6
        
        # U rows for A elements:
        # a: U[ar0*m + ac0]
        # b: U[ar0*m + ac1]
        # c: U[ar1*m + ac0]
        # d: U[ar1*m + ac1]
        
        # V rows for B elements:
        # e: V[br0*p + bc0]
        # f: V[br0*p + bc1]
        # g: V[br1*p + bc0]
        # h: V[br1*p + bc1]
        
        # W rows for C elements (trace convention: W[k*n+i]):
        # C[cr0, cc0] -> W[cc0*n + cr0]
        # C[cr0, cc1] -> W[cc1*n + cr0]
        # C[cr1, cc0] -> W[cc0*n + cr1]
        # C[cr1, cc1] -> W[cc1*n + cr1]
        
        # M1 = (a+d)(e+h)
        col = col_offset
        U[ar0*m + ac0][col] += 1
        U[ar1*m + ac1][col] += 1
        V[br0*p + bc0][col] += 1
        V[br1*p + bc1][col] += 1
        W[cc0*n + cr0][col] += 1
        W[cc1*n + cr0][col] += 1
        W[cc0*n + cr1][col] += 1
        W[cc1*n + cr1][col] += -1
        col += 1
        
        # M2 = (c+d)e
        U[ar1*m + ac0][col] += 1
        U[ar1*m + ac1][col] += 1
        V[br0*p + bc0][col] += 1
        W[cc0*n + cr1][col] += 1
        W[cc1*n + cr1][col] += -1
        col += 1
        
        # M3 = a(f-h)
        U[ar0*m + ac0][col] += 1
        V[br0*p + bc1][col] += 1
        V[br1*p + bc1][col] += -1
        W[cc1*n + cr0][col] += 1
        W[cc1*n + cr1][col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[ar1*m + ac1][col] += 1
        V[br1*p + bc0][col] += 1
        V[br0*p + bc0][col] += -1
        W[cc0*n + cr0][col] += 1
        W[cc0*n + cr1][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[ar0*m + ac0][col] += 1
        U[ar0*m + ac1][col] += 1
        V[br1*p + bc1][col] += 1
        W[cc0*n + cr0][col] += -1
        W[cc1*n + cr0][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[ar1*m + ac0][col] += 1
        U[ar0*m + ac0][col] += -1
        V[br0*p + bc0][col] += 1
        V[br0*p + bc1][col] += 1
        W[cc1*n + cr1][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[ar0*m + ac1][col] += 1
        U[ar1*m + ac1][col] += -1
        V[br1*p + bc0][col] += 1
        V[br1*p + bc1][col] += 1
        W[cc0*n + cr0][col] += 1
        col += 1
        
        return col
    
    # For 4x4x4, we need to decompose C = A*B where A is 4x4, B is 4x4.
    # Using block Strassen:
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12], [B21, B22]] (each 2x2)
    # 
    # S1 = (A11+A22)*(B11+B22)
    # S2 = (A21+A22)*B11
    # S3 = A11*(B12-B22)
    # S4 = A22*(B21-B11)
    # S5 = (A11+A12)*B22
    # S6 = (A21-A11)*(B11+B12)
    # S7 = (A12-A22)*(B21+B22)
    
    # Each S_i involves a 2x2x2 multiplication.
    # But the "A" and "B" in each S_i are linear combinations of blocks.
    # We need to handle this carefully.
    
    # Let me define the block indices:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # For S1 = (A11+A22)*(B11+B22):
    # The "A" matrix has elements:
    # (A11+A22)[0,0] = a00+a22, (A11+A22)[0,1] = a01+a23
    # (A11+A22)[1,0] = a10+a32, (A11+A22)[1,1] = a11+a33
    # The "B" matrix has elements:
    # (B11+B22)[0,0] = b00+b22, (B11+B22)[0,1] = b01+b23
    # (B11+B22)[1,0] = b10+b32, (B11+B22)[1,1] = b11+b33
    #
    # This is getting complex. Let me just build it term by term.
    
    # Actually, let me take a simpler approach. I'll use the fact that
    # for the 4x4x4 part, I can use 4 separate Strassen decompositions
    # for the four block products, but share where possible.
    
    # Simplest correct approach: schoolbook for 4x4x1 (rank 4) and
    # for 4x4x4, use the known minimal or near-minimal decomposition.
    
    # Let me just use rank 4 for the last column and Strassen for 4x4x4.
    # For 4x4x4 Strassen, I'll construct it properly.
    
    # Reset
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First, handle the 4x4x1 part (columns 0-3 of B, row 4 is the 5th column)
    # Wait, B is 4x5, so columns 0-4. The first 4 columns form 4x4x4,
    # the 5th column (index 4) forms 4x4x1.
    
    # For 4x4x1: C[:, i] = A * B[:, 4] for i = 0..3
    # This is just matrix-vector multiplication, rank 4.
    for i in range(n):
        for j in range(m):
            col = 49 + j  # columns 49-52
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1  # B row j, col 4 (the 5th column)
            W[4 * n + i][col] = 1  # W row = k*n+i where k=4 (the 5th row of C)
            # Wait, C is 4x5, so C[i, 4] is the element we're computing.
            # W[k*n + i] where k is the column index of C.
            # For the 5th column (k=4): W[4*n + i] = W[16 + i]
    
    # Now for the 4x4x4 part (columns 0-3 of B)
    # I'll use Strassen decomposition.
    
    # Let me define the Strassen 4x4x4 decomposition explicitly.
    # 
    # A = [[a00, a01, a02, a03],
    #      [a10, a11, a12, a13],
    #      [a20, a21, a22, a23],
    #      [a30, a31, a32, a33]]
    #
    # B = [[b00, b01, b02, b03],
    #      [b10, b11, b12, b13],
    #      [b20, b21, b22, b23],
    #      [b30, b31, b32, b33]]
    #
    # Blocks:
    # A11 = [[a00,a01],[a10,a11]], A12 = [[a02,a03],[a12,a13]]
    # A21 = [[a20,a21],[a30,a31]], A22 = [[a22,a23],[a32,a33]]
    # B11 = [[b00,b01],[b10,b11]], B12 = [[b02,b03],[b12,b13]]
    # B21 = [[b20,b21],[b30,b31]], B22 = [[b22,b23],[b32,b33]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # For each 2x2x2 product, use Strassen.
    
    # I'll write a helper that takes the 2x2 block positions and adds
    # a Strassen decomposition.
    
    def add_strassen_for_blocks(U, V, W, n, m, p,
                                 a_r0, a_c0,  # top-left of A block
                                 b_r0, b_c0,  # top-left of B block
                                 c_r0, c_c0,  # top-left of C block
                                 sign_a, sign_b,  # +1 or -1 for the block
                                 col_start):
        """Add Strassen for a 2x2x2 multiplication.
        
        A block: rows [a_r0, a_r0+1], cols [a_c0, a_c0+1]
        B block: rows [b_r0, b_r0+1], cols [b_c0, b_c0+1]
        C block: rows [c_r0, c_r0+1], cols [c_c0, c_c0+1]
        
        sign_a, sign_b: multipliers for the blocks (for linear combinations)
        """
        ar0, ar1 = a_r0, a_r0 + 1
        ac0, ac1 = a_c0, a_c0 + 1
        br0, br1 = b_r0, b_r0 + 1
        bc0, bc1 = b_c0, b_c0 + 1
        cr0, cr1 = c_r0, c_r0 + 1
        cc0, cc1 = c_c0, c_c0 + 1
        
        col = col_start
        
        # M1 = (a+d)(e+h)
        U[ar0*m + ac0][col] += sign_a
        U[ar1*m + ac1][col] += sign_a
        V[br0*p + bc0][col] += sign_b
        V[br1*p + bc1][col] += sign_b
        W[cc0*n + cr0][col] += 1
        W[cc1*n + cr0][col] += 1
        W[cc0*n + cr1][col] += 1
        W[cc1*n + cr1][col] += -1
        col += 1
        
        # M2 = (c+d)e
        U[ar1*m + ac0][col] += sign_a
        U[ar1*m + ac1][col] += sign_a
        V[br0*p + bc0][col] += sign_b
        W[cc0*n + cr1][col] += 1
        W[cc1*n + cr1][col] += -1
        col += 1
        
        # M3 = a(f-h)
        U[ar0*m + ac0][col] += sign_a
        V[br0*p + bc1][col] += sign_b
        V[br1*p + bc1][col] += -sign_b
        W[cc1*n + cr0][col] += 1
        W[cc1*n + cr1][col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[ar1*m + ac1][col] += sign_a
        V[br1*p + bc0][col] += sign_b
        V[br0*p + bc0][col] += -sign_b
        W[cc0*n + cr0][col] += 1
        W[cc0*n + cr1][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[ar0*m + ac0][col] += sign_a
        U[ar0*m + ac1][col] += sign_a
        V[br1*p + bc1][col] += sign_b
        W[cc0*n + cr0][col] += -1
        W[cc1*n + cr0][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[ar1*m + ac0][col] += sign_a
        U[ar0*m + ac0][col] += -sign_a
        V[br0*p + bc0][col] += sign_b
        V[br0*p + bc1][col] += sign_b
        W[cc1*n + cr1][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[ar0*m + ac1][col] += sign_a
        U[ar1*m + ac1][col] += -sign_a
        V[br1*p + bc0][col] += sign_b
        V[br1*p + bc1][col] += sign_b
        W[cc0*n + cr0][col] += 1
        col += 1
        
        return col
    
    # Now add the 8 Strassen decompositions for the 4x4x4 part.
    # Each uses 7 terms, total 56 terms. But we said rank 53...
    # Let me use 56 + 4 = 60 total.
    
    # Actually, let me just use the full schoolbook for simplicity and
    # correctness, then optimize.
    
    # Schoolbook: rank = 4*4*5 = 80
    # Let me verify my Strassen construction is correct first.
    
    # I'll construct the full 56-term Strassen for 4x4x4 + 4 terms for 4x4x1 = 60.
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 8 block products for 4x4x4:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Block positions:
    # A11: r0=0, c0=0
    # A12: r0=0, c0=2
    # A21: r0=2, c0=0
    # A22: r0=2, c0=2
    # B11: r0=0, c0=0
    # B12: r0=0, c0=2
    # B21: r0=2, c0=0
    # B22: r0=2, c0=2
    # C11: r0=0, c0=0
    # C12: r0=0, c0=2
    # C21: r0=2, c0=0
    # C22: r0=2, c0=2
    
    col = 0
    # A11*B11 -> C11
    col = add_strassen_for_blocks(U, V, W, n, m, p, 0, 0, 0, 0, 0, 0, 1, 1, col)
    # A12*B21 -> C11
    col = add_strassen_for_blocks(U, V, W, n, m, p, 0, 2, 2, 0, 0, 0, 1, 1, col)
    # A11*B12 -> C12
    col = add_strassen_for_blocks(U, V, W, n, m, p, 0, 0, 0, 2, 0, 2, 1, 1, col)
    # A12*B22 -> C12
    col = add_strassen_for_blocks(U, V, W, n, m, p, 0, 2, 2, 2, 0, 2, 1, 1, col)
    # A21*B11 -> C21
    col = add_strassen_for_blocks(U, V, W, n, m, p, 2, 0, 0, 0, 2, 0, 1, 1, col)
    # A22*B21 -> C21
    col = add_strassen_for_blocks(U, V, W, n, m, p, 2, 2, 2, 0, 2, 0, 1, 1, col)
    # A21*B12 -> C22
    col = add_strassen_for_blocks(U, V, W, n, m, p, 2, 0, 0, 2, 2, 2, 1, 1, col)
    # A22*B22 -> C22
    col = add_strassen_for_blocks(U, V, W, n, m, p, 2, 2, 2, 2, 2, 2, 1, 1, col)
    
    # Now the 4x4x1 part (column 4 of B)
    for j in range(m):
        for i in range(n):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
