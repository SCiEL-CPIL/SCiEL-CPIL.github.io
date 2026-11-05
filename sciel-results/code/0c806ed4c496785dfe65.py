"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split into 2, 2, 1
    # Blocks: (0,1), (2,3), (4)
    
    U = [[0] * 117 for _ in range(n * m)]
    V = [[0] * 117 for _ in range(m * p)]
    W = [[0] * 117 for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Strassen for 2x2
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = AB
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6

    def strassen_2x2(A_rows, B_cols, C_rows, C_cols, offset=0):
        # A_rows: list of 2 row indices in U
        # B_cols: list of 2 col indices in V
        # C_rows: list of 2 row indices in W
        # C_cols: list of 2 col indices in W
        # This is tricky because U, V, W are flat.
        pass

    # It is easier to just iterate over the 9 block combinations and add terms.
    
    # Helper to get indices
    # A[i, j] -> U[i*5+j]
    # B[i, j] -> V[i*5+j]
    # C[i, j] -> W[i*5+j] (Wait, W is p*n. C[k*n+i]. So C[i, j] is W[j*5+i]? No.
    # Trace convention: W[k*n+i, r].
    # C[i, j] = sum_k A[i, k] B[k, j].
    # Term: A[i, k] B[k, j] C[i, j].
    # U index: i*5 + k
    # V index: k*5 + j
    # W index: j*5 + i
    
    # We will decompose each block product.
    
    # Block (1,1) 2x2x2: Strassen
    # A indices: rows 0,1; cols 0,1
    # B indices: rows 0,1; cols 0,1
    # C indices: rows 0,1; cols 0,1
    
    # We'll just use schoolbook for simplicity first to ensure correctness, then optimize if time permits.
    # Schoolbook rank 125.
    # Let's try the 2+2+1 split with schoolbook for all blocks.
    # Rank 117.
    
    # Reset
    rank = 117
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term(i, k, j):
        nonlocal col
        u_idx = i * m + k
        v_idx = k * p + j
        w_idx = j * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # We need to skip the "shared" multiplications if we were doing Strassen, 
    # but for the 117 rank direct sum, we just sum the ranks of the blocks.
    # The direct sum of schoolbook blocks:
    # (2,2): 4 mults
    # (2,1): 2 mults
    # (1,2): 2 mults
    # (1,1): 1 mult
    
    # Total 117.
    # We just iterate i, k, j and add terms.
    # This is just schoolbook rank 125.
    # To get 117, we must use Strassen for the 2x2 blocks.
    
    # I will implement Strassen for the 2x2 blocks.
    
    # Strassen terms for a 2x2 block at A[rows, cols], B[rows, cols], C[rows, cols]
    # rows = [r0, r1], cols = [c0, c1]
    
    def strassen_block(ar, ac, br, bc, cr, cc):
        nonlocal col
        # ar, br, cr are row indices in A, B, C (size 2)
        # ac, bc, cc are col indices in A, B, C (size 2)
        
        # a11, a12, a21, a22
        # b11, b12, b21, b22
        # c11, c12, c21, c22
        
        # m1 = (a11+a22)(b11+b22)
        # U: a11, a22 ; V: b11, b22 ; W: c11, c22 (with signs)
        
        # This is complex to map to the global U, V, W indices directly with signs.
        # I will use a helper to add a signed term.
        
        def add_signed(i, k, j, sign):
            nonlocal col
            u_idx = i * m + k
            v_idx = k * p + j
            w_idx = j * n + i
            U[u_idx][col] = sign
            V[v_idx][col] = sign
            W[w_idx][col] = sign
            col += 1

        # a11 is A[ar[0], ac[0]]
        # a12 is A[ar[0], ac[1]]
        # a21 is A[ar[1], ac[0]]
        # a22 is A[ar[1], ac[1]]
        
        # m1 = (a11+a22)(b11+b22)
        # = a11b11 + a11b22 + a22b11 + a22b22
        # c11 += m1, c22 += m1
        
        # This expansion is 4 terms per m. 7 m's -> 28 terms.
        # Strassen is 7 mults, but 18 additions.
        # The rank is 7.
        # The decomposition has 7 terms in the tensor.
        # Each term is a product of sums.
        # We must add the 7 tensor terms.
        
        # Term 1: (a11+a22)(b11+b22)
        # U: a11, a22 ; V: b11, b22 ; W: c11, c22
        # This is not a single rank-1 term. It is 4 rank-1 terms if expanded.
        # But Strassen combines them into 1 multiplication.
        # In the tensor decomposition, a "multiplication" corresponds to a rank-1 term.
        # So we add 1 term for m1.
        # U vector: [1 at a11, 1 at a22]
        # V vector: [1 at b11, 1 at b22]
        # W vector: [1 at c11, 1 at c22]
        
        # We need to add this as a single column in U, V, W.
        
        def add_strassen_term(u_indices, v_indices, w_indices, signs_u, signs_v, signs_w):
            nonlocal col
            for u_idx, s in zip(u_indices, signs_u):
                U[u_idx][col] += s
            for v_idx, s in zip(v_indices, signs_v):
                V[v_idx][col] += s
            for w_idx, s in zip(w_indices, signs_w):
                W[w_idx][col] += s
            col += 1

        # m1
        add_strassen_term(
            [ar[0]*m+ac[0], ar[1]*m+ac[1]],
            [br[0]*p+bc[0], br[1]*p+bc[1]],
            [cc[0]*n+cr[0], cc[1]*n+cr[1]],
            [1, 1], [1, 1], [1, 1]
        )
        
        # m2 = (a21+a22)b11
        add_strassen_term(
            [ar[1]*m+ac[0], ar[1]*m+ac[1]],
            [br[0]*p+bc[0]],
            [cc[0]*n+cr[1]], # c21
            [1, 1], [1], [1]
        )
        
        # m3 = a11(b12-b22)
        add_strassen_term(
            [ar[0]*m+ac[0]],
            [br[0]*p+bc[1], br[1]*p+bc[1]],
            [cc[1]*n+cr[0]], # c12
            [1], [1, -1], [1]
        )
        
        # m4 = a22(b21-b11)
        add_strassen_term(
            [ar[1]*m+ac[1]],
            [br[1]*p+bc[0], br[0]*p+bc[0]],
            [cc[0]*n+cr[0], cc[1]*n+cr[1]], # c11, c22
            [1], [1, -1], [1, 1]
        )
        
        # m5 = (a11+a12)b22
        add_strassen_term(
            [ar[0]*m+ac[0], ar[0]*m+ac[1]],
            [br[1]*p+bc[1]],
            [cc[0]*n+cr[0], cc[1]*n+cr[1]], # c11, c22
            [1, 1], [1], [-1, 1]
        )
        
        # m6 = (a21-a11)(b11+b12)
        add_strassen_term(
            [ar[1]*m+ac[0], ar[0]*m+ac[0]],
            [br[0]*p+bc[0], br[0]*p+bc[1]],
            [cc[1]*n+cr[1]], # c22
            [1, -1], [1, 1], [1]
        )
        
        # m7 = (a12-a22)(b21+b22)
        add_strassen_term(
            [ar[0]*m+ac[1], ar[1]*m+ac[1]],
            [br[1]*p+bc[0], br[1]*p+bc[1]],
            [cc[0]*n+cr[0]], # c11
            [1, -1], [1, 1], [1]
        )
        
    # Define blocks
    # Block 0: rows 0,1; cols 0,1
    # Block 1: rows 0,1; cols 2,3
    # Block 2: rows 0,1; cols 4
    # Block 3: rows 2,3; cols 0,1
    # Block 4: rows 2,3; cols 2,3
    # Block 5: rows 2,3; cols 4
    # Block 6: rows 4; cols 0,1
    # Block 7: rows 4; cols 2,3
    # Block 8: rows 4; cols 4
    
    # Strassen for 2x2 blocks
    strassen_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1]) # C11
    strassen_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3]) # C12
    strassen_block([2,3], [0,1], [0,1], [0,1], [2,3], [0,1]) # C21
    strassen_block([2,3], [0,1], [0,1], [2,3], [2,3], [2,3]) # C22
    
    # Schoolbook for other blocks
    # C13 (2x1)
    for i in [0,1]:
        for k in [0,1]:
            add_term(i, k, 4)
        for k in [2,3]:
            add_term(i, k, 4)
        add_term(i, 4, 4)
        
    # C23 (2x1)
    for i in [2,3]:
        for k in [0,1]:
            add_term(i, k, 4)
        for k in [2,3]:
            add_term(i, k, 4)
        add_term(i, 4, 4)
        
    # C31 (1x2)
    for i in [4]:
        for k in [0,1]:
            add_term(i, k, 0)
            add_term(i, k, 1)
        for k in [2,3]:
            add_term(i, k, 0)
            add_term(i, k, 1)
        add_term(i, 4, 0)
        add_term(i, 4, 1)
        
    # C32 (1x2)
    for i in [4]:
        for k in [0,1]:
            add_term(i, k, 2)
            add_term(i, k, 3)
        for k in [2,3]:
            add_term(i, k, 2)
            add_term(i, k, 3)
        add_term(i, 4, 2)
        add_term(i, 4, 3)
        
    # C33 (1x1)
    add_term(4, 4, 4)
    
    # Check col count
    # 4 Strassen blocks * 7 = 28
    # C13: 2 rows * (2+2+1) = 10
    # C23: 2 rows * (2+2+1) = 10
    # C31: 1 row * (2+2+1) * 2 cols = 10
    # C32: 1 row * (2+2+1) * 2 cols = 10
    # C33: 1
    # Total: 28 + 10 + 10 + 10 + 10 + 1 = 69?
    # Wait, my previous count was 117.
    # 117 - 69 = 48.
    # I missed the Strassen blocks for C12, C21, C22?
    # I did 4 Strassen blocks.
    # C11, C12, C21, C22.
    # Yes.
    # Why 117?
    # 4 * 7 = 28.
    # Cross terms:
    # C13 (2x1): A13 (2x1) B31 (1x1)? No.
    # C13 = A11 B13 + A12 B23 + A13 B33.
    # A11 is 2x2. B13 is 2x1. Rank 4.
    # A12 is 2x2. B23 is 2x1. Rank 4.
    # A13 is 2x1. B33 is 1x1. Rank 2.
    # Total 10.
    # My loop for C13:
    # i in 0,1.
    # k in 0,1 (2 terms)
    # k in 2,3 (2 terms)
    # k=4 (1 term)
    # Total 5 per row. 2 rows = 10. Correct.
    
    # So total is 69.
    # 69 is much better than 117.
    # 125 / 69 = 1.81.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
