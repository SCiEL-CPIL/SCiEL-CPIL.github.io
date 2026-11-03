"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition:
    # A (3x4) is split into A_top (2x4) and A_bottom (1x4).
    # A_top is split into A_top_left (2x2) and A_top_right (2x2).
    # A_bottom is split into A_bottom_left (1x2) and A_bottom_right (1x2).
    # B (4x6) is split into B_top (2x6) and B_bottom (2x6).
    # B_top and B_bottom are each split into 3 blocks of (2x2).
    
    # Rank contributions:
    # A_top * B_top: 2x2 by 2x6 -> 3 blocks of 2x2x2 (Rank 7 each) -> 21
    # A_top * B_bottom: 2x2 by 2x6 -> 3 blocks of 2x2x2 (Rank 7 each) -> 21
    # A_bottom * B_top: 1x2 by 2x6 -> 3 blocks of 1x2x2 (Rank 2 each) -> 6
    # A_bottom * B_bottom: 1x2 by 2x6 -> 3 blocks of 1x2x2 (Rank 2 each) -> 6
    # Total Rank: 21 + 21 + 6 + 6 = 54.

    rank = 54
    U = [[0] * rank for _ in range(n * m)] # 12 x 54
    V = [[0] * rank for _ in range(m * p)] # 24 x 54
    W = [[0] * rank for _ in range(p * n)] # 18 x 54
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # Indices in U, V, W correspond to the block positions.
    # U is indexed by [i*m + j]. A is 3x4.
    # V is indexed by [j*p + k]. B is 4x6.
    # W is indexed by [k*n + i]. C is 6x3.
    
    def add_strassen_222(u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i):
        nonlocal col
        # Strassen decomposition for C = A*B where A, B, C are 2x2.
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
        # m1 = (a+d)(e+h)
        # m2 = d(f-h)
        # m3 = a(e-f)
        # m4 = c(g+e)
        # m5 = (a+b)h
        # m6 = (c-d)(f+g)
        # m7 = (b-c)(e+h)
        # i = m1 + m4 - m5 + m7
        # j = m3 + m5
        # k = m2 + m4
        # l = m1 - m3 + m2 - m6
        
        # We need to map the local 2x2 indices to the global U, V, W indices.
        # u_idx = u_offset_i + di, u_offset_j + dj
        # v_idx = v_offset_j + dj, v_offset_k + dk
        # w_idx = w_offset_k + dk, w_offset_i + di
        
        # m1: (a+d)(e+h)
        # U: (0,0), (1,1)
        # V: (0,0), (1,1)
        # W: (0,0), (1,1) with sign +
        U[u_offset_i + 0][col] = 1
        U[u_offset_i + 1][col] = 1
        V[v_offset_j + 0][col] = 1
        V[v_offset_j + 1][col] = 1
        W[w_offset_k + 0][col] = 1
        W[w_offset_k + 1][col] = 1
        col += 1
        
        # m2: d(f-h)
        # U: (1,1)
        # V: (0,1), (1,0)
        # W: (0,1), (1,0) with sign + for (0,1), - for (1,0)?
        # C[0,1] += m2, C[1,0] += m2?
        # l = m1 - m3 + m2 - m6. k = m2 + m4.
        # So m2 contributes to C[1,0] and C[1,1]?
        # Strassen formulas:
        # C[0,0] = m1 + m4 - m5 + m7
        # C[0,1] = m3 + m5
        # C[1,0] = m2 + m4
        # C[1,1] = m1 - m3 + m2 - m6
        
        # m2 = d * (f - h).
        # U[1,1] = 1.
        # V[0,1] = 1, V[1,0] = -1.
        # W[1,0] = 1, W[1,1] = 1.
        U[u_offset_i + 1][col] = 1
        V[v_offset_j + 0][col] = 1
        V[v_offset_j + 1][col] = -1
        W[w_offset_k + 0][col] = 1
        W[w_offset_k + 1][col] = 1
        col += 1
        
        # m3: a(e-f)
        # U[0,0] = 1.
        # V[0,0] = 1, V[0,1] = -1.
        # W[0,1] = 1, W[1,1] = -1.
        U[u_offset_i + 0][col] = 1
        V[v_offset_j + 0][col] = 1
        V[v_offset_j + 1][col] = -1
        W[w_offset_k + 1][col] = 1
        W[w_offset_k + 0][col] = -1
        col += 1
        
        # m4: c(g+e)
        # U[1,0] = 1.
        # V[1,0] = 1, V[0,0] = 1.
        # W[0,0] = 1, W[1,0] = 1.
        U[u_offset_i + 1][col] = 1
        V[v_offset_j + 1][col] = 1
        V[v_offset_j + 0][col] = 1
        W[w_offset_k + 0][col] = 1
        W[w_offset_k + 1][col] = 1
        col += 1
        
        # m5: (a+b)h
        # U[0,0]=1, U[0,1]=1.
        # V[1,1]=1.
        # W[0,0]=-1, W[0,1]=1.
        U[u_offset_i + 0][col] = 1
        U[u_offset_i + 1][col] = 1
        V[v_offset_j + 1][col] = 1
        W[w_offset_k + 0][col] = -1
        W[w_offset_k + 1][col] = 1
        col += 1
        
        # m6: (c-d)(f+g)
        # U[1,0]=1, U[1,1]=-1.
        # V[0,1]=1, V[1,0]=1.
        # W[1,1]=-1.
        U[u_offset_i + 1][col] = 1
        U[u_offset_i + 0][col] = -1 # c is (1,0), d is (0,1) in 0-indexed?
        # Wait, A = [[a,b],[c,d]].
        # a=(0,0), b=(0,1), c=(1,0), d=(1,1).
        # So c-d is U[1,0]=1, U[1,1]=-1.
        V[v_offset_j + 0][col] = 1
        V[v_offset_j + 1][col] = 1
        W[w_offset_k + 1][col] = -1
        col += 1
        
        # m7: (b-c)(e+h)
        # U[0,1]=1, U[1,0]=-1.
        # V[0,0]=1, V[1,0]=1.
        # W[0,0]=1.
        U[u_offset_i + 0][col] = 1
        U[u_offset_i + 1][col] = -1
        V[v_offset_j + 0][col] = 1
        V[v_offset_j + 1][col] = 1
        W[w_offset_k + 0][col] = 1
        col += 1

    def add_dot_122(u_offset_i, u_offset_j, v_offset_j, v_offset_k, w_offset_k, w_offset_i):
        nonlocal col
        # C = A * B where A is 1x2, B is 2x2.
        # A = [a, b]. B = [[e, f], [g, h]].
        # C = [ae+bg, af+bh].
        # Two multiplications:
        # 1. a * e -> C[0]
        # 2. b * g -> C[0]
        # 3. a * f -> C[1]
        # 4. b * h -> C[1]
        # Wait, rank is 2?
        # C[0] = ae + bg.
        # C[1] = af + bh.
        # We can compute:
        # m1 = a * (e+f) ? No.
        # If we use 2 multiplications:
        # m1 = a * e + b * g ? No, that's not a product of linear forms in a single variable set?
        # Rank 2 means we can write C as sum of 2 rank-1 tensors.
        # T1: a * e -> C[0].
        # T2: b * g -> C[0].
        # T3: a * f -> C[1].
        # T4: b * h -> C[1].
        # This is 4 multiplications.
        # Can we do it in 2?
        # m1 = (a+b)(e+g) = ae + ag + be + bg.
        # m2 = a * f + b * h ?
        # No, m2 must be a product of linear forms.
        # It turns out Rank(1x2x2) is 2.
        # Decomposition:
        # m1 = a * (e + f) ? No.
        # Let's check the rank of 1x2x2.
        # T = a1 e1 + a1 f1 + a2 g1 + a2 h1.
        # This is the tensor of bilinear form.
        # It is known that Rank(1x2x2) = 2.
        # One decomposition:
        # m1 = a * e + b * g ? No.
        # Actually, we can use:
        # m1 = a * (e + g)
        # m2 = b * (f + h)
        # C[0] = m1 + ?
        # This doesn't work directly.
        
        # Let's use 4 multiplications to be safe, or find the 2-mult decomp.
        # If Rank is 2, we need:
        # C[0] = u1 a v1 + u2 a v2 ?
        # Actually, for 1x2x2, we can just use schoolbook (4 mults) if we don't know the 2-mult decomp.
        # But 4*6 = 24. Total 42+24=66. Still better than 72.
        # Let's use 4 mults for simplicity and correctness.
        
        # 1. a * e
        U[u_offset_i + 0][col] = 1
        V[v_offset_j + 0][col] = 1
        W[w_offset_k + 0][col] = 1
        col += 1
        # 2. b * g
        U[u_offset_i + 1][col] = 1
        V[v_offset_j + 1][col] = 1
        W[w_offset_k + 0][col] = 1
        col += 1
        # 3. a * f
        U[u_offset_i + 0][col] = 1
        V[v_offset_j + 0][col] = 1
        W[w_offset_k + 1][col] = 1
        col += 1
        # 4. b * h
        U[u_offset_i + 1][col] = 1
        V[v_offset_j + 1][col] = 1
        W[w_offset_k + 1][col] = 1
        col += 1

    # Block 1: A_top (rows 0,1) * B_top (cols 0,1 of B? No, B is 4x6. Rows 0,1 of B)
    # A_top is 2x4. Split into A_top_left (2x2, cols 0,1) and A_top_right (2x2, cols 2,3).
    # B_top is 2x6. Split into 3 blocks of 2x2.
    
    # A_top_left (2x2) is at A[0:2, 0:2].
    # B_top blocks are at B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6].
    
    # C_top_left (2x2) is at C[0:2, 0:2].
    add_strassen_222(0, 0, 0, 0, 0, 0)
    add_strassen_222(0, 2, 0, 2, 2, 0)
    add_strassen_222(0, 4, 0, 4, 4, 0)
    
    # A_top_right (2x2) is at A[0:2, 2:4].
    # B_bottom blocks are at B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6].
    # C_top_right (2x2) is at C[0:2, 0:2]? No, C is 6x3.
    # C indices: k*n + i.
    # C[0:2, 0:2] corresponds to k=0,1 and i=0,1.
    # The multiplication A_top_right * B_bottom contributes to C[0:2, 0:2].
    add_strassen_222(2, 0, 2, 0, 0, 0)
    add_strassen_222(2, 2, 2, 2, 2, 0)
    add_strassen_222(2, 4, 2, 4, 4, 0)
    
    # A_bottom (row 2) is 1x4. Split into A_bottom_left (1x2, cols 0,1) and A_bottom_right (1x2, cols 2,3).
    # B_top blocks.
    # C_bottom (1x2) is at C[0:2, 2:2]? No, C is 6x3.
    # i=2 is the third row of C.
    # C[k, 2] for k=0..5.
    # So C_bottom is C[0:6, 2:3].
    
    # A_bottom_left (1x2) at A[2, 0:2].
    # B_top blocks at B[0:2, 0:2], etc.
    add_dot_122(6, 0, 0, 0, 0, 2) # u_offset_i=6 (row 2 of A, 0-indexed 2? A is 3x4. i=2. u_idx = 2*4+0 = 8? No.
    # U is (n*m, rank). n=3, m=4.
    # i is row of A (0..2). j is col of A (0..3).
    # u_idx = i*m + j.
    # A_bottom is i=2.
    # A_bottom_left is j=0,1.
    # So u_offset_i = 2*4 = 8.
    
    # Let's re-verify indices.
    # U: (12, rank). Index i*m+j.
    # V: (24, rank). Index j*p+k. j is row of B (0..3). k is col of B (0..5).
    # W: (18, rank). Index k*n+i. k is row of C (0..5). i is col of C (0..2).
    
    # A_top: i=0,1.
    # A_top_left: j=0,1. u_idx = 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5.
    # A_top_right: j=2,3. u_idx = 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7.
    
    # A_bottom: i=2.
    # A_bottom_left: j=0,1. u_idx = 2*4+0=8, 2*4+1=9.
    # A_bottom_right: j=2,3. u_idx = 2*4+2=10, 2*4+3=11.
    
    # B_top: j=0,1.
    # B_top blocks: k=0,1; k=2,3; k=4,5.
    # v_idx = j*p+k.
    # Block 1: j=0,1; k=0,1. v_idx = 0*6+0=0, 0*6+1=1, 1*6+0=6, 1*6+1=7.
    # Block 2: j=0,1; k=2,3. v_idx = 0*6+2=2, 0*6+3=3, 1*6+2=8, 1*6+3=9.
    # Block 3: j=0,1; k=4,5. v_idx = 0*6+4=4, 0*6+5=5, 1*6+4=10, 1*6+5=11.
    
    # B_bottom: j=2,3.
    # Block 1: j=2,3; k=0,1. v_idx = 2*6+0=12, 2*6+1=13, 3*6+0=18, 3*6+1=19.
    # Block 2: j=2,3; k=2,3. v_idx = 2*6+2=14, 2*6+3=15, 3*6+2=20, 3*6+3=21.
    # Block 3: j=2,3; k=4,5. v_idx = 2*6+4=16, 2*6+5=17, 3*6+4=22, 3*6+5=23.
    
    # C_top: i=0,1.
    # C_top blocks: k=0,1; k=2,3; k=4,5.
    # w_idx = k*n+i.
    # Block 1: k=0,1; i=0,1. w_idx = 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4.
    # Block 2: k=2,3; i=0,1. w_idx = 2*3+0=6, 2*3+1=7, 3*3+0=9, 3*3+1=10.
    # Block 3: k=4,5; i=0,1. w_idx = 4*3+0=12, 4*3+1=13, 5*3+0=15, 5*3+1=16.
    
    # C_bottom: i=2.
    # C_bottom blocks: k=0,1; k=2,3; k=4,5.
    # w_idx = k*n+2.
    # Block 1: k=0,1; i=2. w_idx = 0*3+2=2, 1*3+2=5.
    # Block 2: k=2,3; i=2. w_idx = 2*3+2=8, 3*3+2=11.
    # Block 3: k=4,5; i=2. w_idx = 4*3+2=14, 5*3+2=17.

    # Reset col to 0 and re-run with correct offsets.
    col = 0
    
    # A_top_left * B_top blocks
    add_strassen_222(0, 0, 0, 0, 0, 0)
    add_strassen_222(0, 2, 0, 2, 6, 0) # Wait, v_offset_j=0, v_offset_k=2. w_offset_k=2, w_offset_i=0.
    # My helper function takes offsets for the top-left corner of the 2x2 block.
    # u_offset_i, u_offset_j: top-left of A block.
    # v_offset_j, v_offset_k: top-left of B block.
    # w_offset_k, w_offset_i: top-left of C block.
    
    # A_top_left: u_offset_i=0, u_offset_j=0.
    # B_top Block 1: v_offset_j=0, v_offset_k=0.
    # C_top Block 1: w_offset_k=0, w_offset_i=0.
    add_strassen_222(0, 0, 0, 0, 0, 0)
    
    # A_top_left: u_offset_i=0, u_offset_j=0.
    # B_top Block 2: v_offset_j=0, v_offset_k=2.
    # C_top Block 2: w_offset_k=2, w_offset_i=0.
    add_strassen_222(0, 0, 0, 2, 2, 0)
    
    # A_top_left: u_offset_i=0, u_offset_j=0.
    # B_top Block 3: v_offset_j=0, v_offset_k=4.
    # C_top Block 3: w_offset_k=4, w_offset_i=0.
    add_strassen_222(0, 0, 0, 4, 4, 0)
    
    # A_top_right: u_offset_i=0, u_offset_j=2.
    # B_bottom Block 1: v_offset_j=2, v_offset_k=0.
    # C_top Block 1: w_offset_k=0, w_offset_i=0.
    add_strassen_222(0, 2, 2, 0, 0, 0)
    
    # A_top_right: u_offset_i=0, u_offset_j=2.
    # B_bottom Block 2: v_offset_j=2, v_offset_k=2.
    # C_top Block 2: w_offset_k=2, w_offset_i=0.
    add_strassen_222(0, 2, 2, 2, 2, 0)
    
    # A_top_right: u_offset_i=0, u_offset_j=2.
    # B_bottom Block 3: v_offset_j=2, v_offset_k=4.
    # C_top Block 3: w_offset_k=4, w_offset_i=0.
    add_strassen_222(0, 2, 2, 4, 4, 0)
    
    # A_bottom_left: u_offset_i=2, u_offset_j=0.
    # B_top Block 1: v_offset_j=0, v_offset_k=0.
    # C_bottom Block 1: w_offset_k=0, w_offset_i=2.
    add_dot_122(2, 0, 0, 0, 0, 2)
    
    # A_bottom_left: u_offset_i=2, u_offset_j=0.
    # B_top Block 2: v_offset_j=0, v_offset_k=2.
    # C_bottom Block 2: w_offset_k=2, w_offset_i=2.
    add_dot_122(2, 0, 0, 2, 2, 2)
    
    # A_bottom_left: u_offset_i=2, u_offset_j=0.
    # B_top Block 3: v_offset_j=0, v_offset_k=4.
    # C_bottom Block 3: w_offset_k=4, w_offset_i=2.
    add_dot_122(2, 0, 0, 4, 4, 2)
    
    # A_bottom_right: u_offset_i=2, u_offset_j=2.
    # B_bottom Block 1: v_offset_j=2, v_offset_k=0.
    # C_bottom Block 1: w_offset_k=0, w_offset_i=2.
    add_dot_122(2, 2, 2, 0, 0, 2)
    
    # A_bottom_right: u_offset_i=2, u_offset_j=2.
    # B_bottom Block 2: v_offset_j=2, v_offset_k=2.
    # C_bottom Block 2: w_offset_k=2, w_offset_i=2.
    add_dot_122(2, 2, 2, 2, 2, 2)
    
    # A_bottom_right: u_offset_i=2, u_offset_j=2.
    # B_bottom Block 3: v_offset_j=2, v_offset_k=4.
    # C_bottom Block 3: w_offset_k=4, w_offset_i=2.
    add_dot_122(2, 2, 2, 4, 4, 2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
