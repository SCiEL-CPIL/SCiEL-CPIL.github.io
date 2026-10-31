"""4x4 matrix multiplication tensor decomposition using 2-level Strassen."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4 matrix multiplication using a 2-level Strassen algorithm.
    Level 1: Treat 4x4 as 2x2 blocks of 2x2 matrices. Apply Strassen (rank 7).
    Level 2: Treat each 2x2 block product as a standard 2x2 matrix multiplication. Apply Strassen (rank 7).
    Total rank = 7 * 7 = 49.
    """
    n, m, p = DIMENSIONS  # all 4
    
    # Outer Strassen definitions for 2x2 block matrices.
    # Each entry is (coeff, block_row, block_col)
    # U_out[s]: list of (coeff, i, j) for A blocks
    # V_out[s]: list of (coeff, i, j) for B blocks
    # W_out[s]: list of (coeff, i, j) for C blocks (which block of C gets this term)
    
    U_out = [
        [(1, 0, 0), (1, 1, 1)],  # s0: A00 + A11
        [(1, 1, 0), (1, 1, 1)],  # s1: A10 + A11
        [(1, 0, 0)],             # s2: A00
        [(1, 1, 1)],             # s3: A11
        [(1, 0, 0), (1, 0, 1)],  # s4: A00 + A01
        [(1, 1, 0), (-1, 0, 0)],# s5: A10 - A00
        [(1, 0, 1), (-1, 1, 1)]  # s6: A01 - A11
    ]
    
    V_out = [
        [(1, 0, 0), (1, 1, 1)],  # s0: B00 + B11
        [(1, 0, 0)],             # s1: B00
        [(1, 0, 1), (-1, 1, 1)],# s2: B01 - B11
        [(1, 1, 0), (-1, 0, 0)],# s3: B10 - B00
        [(1, 1, 1)],             # s4: B11
        [(1, 0, 0), (1, 0, 1)],  # s5: B00 + B01
        [(1, 1, 0), (1, 1, 1)]   # s6: B10 + B11
    ]
    
    # W_out[s]: which blocks of C are affected by M_s
    # M_s is a 2x2 block matrix.
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    W_out = [
        [(1, 0, 0), (1, 1, 1)],  # s0: C00 += 1, C11 += 1
        [(1, 1, 0), (-1, 1, 1)], # s1: C10 += 1, C11 += -1
        [(1, 0, 1), (1, 1, 1)],  # s2: C01 += 1, C11 += 1
        [(1, 0, 0), (1, 1, 0)],  # s3: C00 += 1, C10 += 1
        [(-1, 0, 0), (1, 0, 1)], # s4: C00 += -1, C01 += 1
        [(1, 1, 1)],             # s5: C11 += 1
        [(1, 0, 0)]              # s6: C00 += 1
    ]
    
    # Inner Strassen definitions for 2x2 matrices.
    # For X, Y in M_2(R), Z = X * Y:
    # m1 = (X00 + X11) * (Y00 + Y11)
    # m2 = (X10 + X11) * Y00
    # m3 = X00 * (Y01 - Y11)
    # m4 = X11 * (Y10 - Y00)
    # m5 = (X00 + X01) * Y11
    # m6 = (X10 - X00) * (Y00 + Y01)
    # m7 = (X01 - X11) * (Y10 + Y11)
    
    # Z00 = m1 + m4 - m5 + m7
    # Z01 = m3 + m5
    # Z10 = m2 + m4
    # Z11 = m1 - m2 + m3 + m6
    
    U_in = [
        [(1, 0, 0), (1, 1, 1)],  # t0
        [(1, 1, 0), (1, 1, 1)],  # t1
        [(1, 0, 0)],             # t2
        [(1, 1, 1)],             # t3
        [(1, 0, 0), (1, 0, 1)],  # t4
        [(1, 1, 0), (-1, 0, 0)],# t5
        [(1, 0, 1), (-1, 1, 1)]  # t6
    ]
    
    V_in = [
        [(1, 0, 0), (1, 1, 1)],  # t0
        [(1, 0, 0)],             # t1
        [(1, 0, 1), (-1, 1, 1)],# t2
        [(1, 1, 0), (-1, 0, 0)],# t3
        [(1, 1, 1)],             # t4
        [(1, 0, 0), (1, 0, 1)],  # t5
        [(1, 1, 0), (1, 1, 1)]   # t6
    ]
    
    W_in = [
        [(1, 0, 0), (1, 1, 1)],  # t0
        [(1, 1, 0), (-1, 1, 1)], # t1
        [(1, 0, 1), (1, 1, 1)],  # t2
        [(1, 0, 0), (1, 1, 0)],  # t3
        [(-1, 0, 0), (1, 0, 1)], # t4
        [(1, 1, 1)],             # t5
        [(1, 0, 0)]              # t6
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    col = 0
    for s in range(7):
        # For each outer term s, we compute M_s = X_s Y_s
        # X_s is a 2x2 matrix of scalars (blocks of A)
        # Y_s is a 2x2 matrix of scalars (blocks of B)
        
        # We need to expand X_s and Y_s into 2x2 matrices of scalars (global elements)
        # X_s[i][j] is a 2x2 matrix formed by linear combos of A's elements.
        # Let's build the 2x2 matrix X_s_local where X_s_local[r][c] is a list of (coeff, global_row, global_col)
        
        X_s_local = [[[] for _ in range(2)] for _ in range(2)]
        Y_s_local = [[[] for _ in range(2)] for _ in range(2)]
        
        # Fill X_s_local
        for (coeff, bi, bj) in U_out[s]:
            # Block (bi, bj) of A corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
            # The block itself is a 2x2 matrix.
            # The elements of this block are at global (2*bi+r, 2*bj+c)
            # These elements contribute to X_s_local[?][?]
            # Wait, X_s is a 2x2 matrix of blocks.
            # X_s[bi][bj] is the block.
            # So X_s[bi][bj] gets coeff * A_block(bi, bj).
            # A_block(bi, bj) is a 2x2 matrix.
            # So for r in 0..1, c in 0..1:
            # X_s[bi][bj][r][c] += coeff * A[2*bi+r, 2*bj+c]
            
            # But X_s is a 2x2 matrix. Its entries are blocks.
            # We are applying inner Strassen to X_s and Y_s.
            # Inner Strassen treats X_s as a 2x2 matrix of "scalars".
            # Here "scalars" are the blocks.
            # So X_s[0][0] is a block (2x2 matrix).
            # Inner Strassen term t uses X_s[0][0], X_s[0][1], etc.
            # So we need to represent X_s[0][0] as a 2x2 matrix of global scalars.
            
            for r in range(2):
                for c in range(2):
                    gr = 2 * bi + r
                    gc = 2 * bj + c
                    # This global element A[gr, gc] contributes to the block X_s[bi][bj]
                    # Specifically, it is at local position (r, c) of that block.
                    # But inner Strassen operates on the 2x2 structure of X_s.
                    # X_s is 2x2 blocks.
                    # Inner Strassen term t uses entries X_s[i][j].
                    # X_s[i][j] is a block.
                    # So for inner term t, we need the "scalar" value at X_s[i][j].
                    # But X_s[i][j] is a 2x2 matrix.
                    # This implies we are doing Strassen on the 4x4 matrix directly?
                    # No.
                    # Let's re-evaluate.
                    
                    # If X_s is a 2x2 matrix of 2x2 blocks, and we apply Strassen to it,
                    # we are treating the blocks as scalars.
                    # The "multiplication" of these scalar blocks is the 2x2 matrix multiplication.
                    # So, for inner Strassen term t, we form linear combinations of the blocks.
                    # e.g., u_t = X_s[0][0] + X_s[1][1].
                    # u_t is a 2x2 matrix (sum of blocks).
                    # v_t is a 2x2 matrix.
                    # The product u_t v_t is a 2x2 matrix product.
                    # This product is computed with rank 7?
                    # No, u_t v_t is ONE 2x2 matrix product.
                    # We use the rank-7 decomposition of 2x2 multiplication to compute u_t v_t.
                    # So, for each outer term s, we have 7 inner terms t.
                    # For each (s, t), we compute u_t v_t.
                    # u_t is a 2x2 matrix (block). v_t is a 2x2 matrix (block).
                    # We decompose u_t v_t using Strassen (rank 7).
                    # So for each (s, t), we need 7 scalar multiplications.
                    # Total 7 * 7 * 7 = 343?
                    
                    # NO.
                    # Strassen for 2x2 matrices (outer) uses 7 multiplications of "scalars".
                    # Here "scalars" are 2x2 matrices.
                    # Multiplication of two 2x2 matrices has rank 7.
                    # So each "scalar" multiplication costs 7.
                    # Total 7 * 7 = 49.
                    
                    # So, for each outer term s, we form u_s and v_s (2x2 matrices of blocks? No, 2x2 matrices of scalars where scalars are blocks).
                    # Wait, u_s is a 2x2 matrix. Its entries are blocks.
                    # Inner Strassen on u_s and v_s?
                    # No. The inner Strassen is applied to the *product* of the blocks.
                    # Let's trace carefully.
                    
                    # Outer: C = sum_s W_s (X_s Y_s).
                    # X_s, Y_s are 2x2 matrices of blocks.
                    # X_s Y_s is a 2x2 matrix of blocks.
                    # (X_s Y_s)_{ij} = sum_k X_s_{ik} Y_s_{kj}.
                    # Each (X_s Y_s)_{ij} is a sum of block products.
                    # In Strassen, we don't compute the sum directly.
                    # We compute 7 block products M_k.
                    # M_k = U_k V_k.
                    # U_k, V_k are blocks (2x2 matrices).
                    # So M_k is a block product.
                    # We compute M_k using inner Strassen (rank 7).
                    
                    # So, for each s (0..6), we have U_s and V_s which are 2x2 matrices (blocks).
                    # Wait, U_s is a block?
                    # In outer Strassen:
                    # M_0 = (A00+A11)(B00+B11).
                    # Here (A00+A11) is a block (2x2 matrix).
                    # (B00+B11) is a block (2x2 matrix).
                    # So U_0 is a 2x2 matrix. V_0 is a 2x2 matrix.
                    # M_0 = U_0 V_0.
                    # We compute M_0 using inner Strassen.
                    # Inner Strassen on U_0 and V_0 (which are 2x2 matrices).
                    # This costs 7 scalar multiplications.
                    
                    # So the structure is:
                    # for s in 0..6:
                    #   U_s = linear combo of A blocks (result is 2x2 matrix)
                    #   V_s = linear combo of B blocks (result is 2x2 matrix)
                    #   for t in 0..6:
                    #     u_t = linear combo of elements of U_s (result is scalar)
                    #     v_t = linear combo of elements of V_s (result is scalar)
                    #     scalar_mult = u_t * v_t
                    #     Add to W based on W_in[t] and W_out[s]
                    
                    pass # Logic confirmed.
        
        # Construct U_s and V_s as 2x2 matrices of global scalar coefficients.
        # U_s[r][c] is a list of (coeff, global_row, global_col)
        
        U_s = [[[] for _ in range(2)] for _ in range(2)]
        V_s = [[[] for _ in range(2)] for _ in range(2)]
        
        for (coeff, bi, bj) in U_out[s]:
            for r in range(2):
                for c in range(2):
                    gr = 2 * bi + r
                    gc = 2 * bj + c
                    U_s[r][c].append((coeff, gr, gc))
                    
        for (coeff, bi, bj) in V_out[s]:
            for r in range(2):
                for c in range(2):
                    gr = 2 * bi + r
                    gc = 2 * bj + c
                    V_s[r][c].append((coeff, gr, gc))
        
        # Now apply inner Strassen to U_s and V_s
        for t in range(7):
            # u_t is a scalar (linear combo of U_s elements)
            # v_t is a scalar (linear combo of V_s elements)
            
            u_t_coeffs = []
            v_t_coeffs = []
            
            for (coeff_in, ri, ci) in U_in[t]:
                for (coeff_s, gr, gc) in U_s[ri][ci]:
                    u_t_coeffs.append((coeff_in * coeff_s, gr, gc))
                    
            for (coeff_in, ri, ci) in V_in[t]:
                for (coeff_s, gr, gc) in V_s[ri][ci]:
                    v_t_coeffs.append((coeff_in * coeff_s, gr, gc))
            
            # Add to global U and V
            for (coeff, gr, gc) in u_t_coeffs:
                U[gr * m + gc][col] += coeff
                
            for (coeff, gr, gc) in v_t_coeffs:
                V[gr * p + gc][col] += coeff
                
            # Add to global W
            # W contribution is sum over W_out[s] and W_in[t]
            # W_out[s] gives block (bi, bj) and coeff_w_out
            # W_in[t] gives local (ri, ci) and coeff_w_in
            # Global W index: (bi, bj) block, local (ri, ci)
            # W[global_row, global_col]
            # W is p x n = 4 x 4.
            # Convention W[k*n+i, r] in task?
            # "W[k*n+i,r]" -> W is (p*n, rank).
            # Index 0..15.
            # k = row, i = col?
            # Task: "output factor follows the official trace convention W[k*n+i,r]"
            # Usually W is p x n.
            # If W is p x n, index is k*n + i where k in 0..p-1, i in 0..n-1.
            # Here p=4, n=4.
            
            for (coeff_out, bi, bj) in W_out[s]:
                for (coeff_in, ri, ci) in W_in[t]:
                    gr = 2 * bi + ri
                    gc = 2 * bj + ci
                    # W[gr * n + gc][col] += coeff_out * coeff_in
                    W[gr * n + gc][col] += coeff_out * coeff_in
                    
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
