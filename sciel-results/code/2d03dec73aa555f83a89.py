"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 56 decomposition using 8 independent 2x2 Strassen multiplications.
    # This is better than the schoolbook rank 64.
    n, m, p = DIMENSIONS
    # We use 8 blocks, each requiring rank 7.
    # Total rank = 8 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The 7 terms for 2x2 Strassen algorithm
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Coefficients for U (A part), V (B part), W (C part)
    # Indices for 2x2:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Term definitions: (u_vec, v_vec, w_vec)
    # u_vec, v_vec, w_vec are length 4 for the 2x2 case.
    
    terms_2x2 = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]),
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 1, -1, 0]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),
    ]
    
    # Map 4x4 block indices to 2x2 local indices
    # Block (bi, bj) in the 2x2 grid of 4x4 matrix
    # bi, bj in {0, 1}
    # Local row/col in 2x2 block: li, lj in {0, 1}
    # Global row i = bi * 2 + li
    # Global col j = bj * 2 + lj
    
    # We need to compute 8 products:
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    
    # For each of the 8 block products P[i,j] = A_block(i, k) * B_block(k, j)
    # We have 8 such products.
    # Let's list them as (A_block_idx, B_block_idx, C_block_idx)
    # A_block_idx is (row_block, col_block)
    # B_block_idx is (row_block, col_block)
    # C_block_idx is (row_block, col_block)
    
    # Product 1: C00 += A00 * B00
    # Product 2: C00 += A01 * B10
    # Product 3: C01 += A00 * B01
    # Product 4: C01 += A01 * B11
    # Product 5: C10 += A10 * B00
    # Product 6: C10 += A11 * B10
    # Product 7: C11 += A10 * B01
    # Product 8: C11 += A11 * B11
    
    products = [
        ((0,0), (0,0), (0,0)),
        ((0,1), (1,0), (0,0)),
        ((0,0), (0,1), (0,1)),
        ((0,1), (1,1), (0,1)),
        ((1,0), (0,0), (1,0)),
        ((1,1), (1,0), (1,0)),
        ((1,0), (0,1), (1,1)),
        ((1,1), (1,1), (1,1)),
    ]
    
    col = 0
    for (ab, bb, cb) in products:
        ab_r, ab_c = ab
        bb_r, bb_c = bb
        cb_r, cb_c = cb
        
        for (u_loc, v_loc, w_loc) in terms_2x2:
            # Update U, V, W
            # A global index: (ab_r * 2 + li, ab_c * 2 + lj)
            # B global index: (bb_r * 2 + li, bb_c * 2 + lj)
            # C global index: (cb_r * 2 + li, cb_c * 2 + lj)
            
            for li in range(2):
                for lj in range(2):
                    val = u_loc[li * 2 + lj]
                    if val != 0:
                        u_idx = (ab_r * 2 + li) * 4 + (ab_c * 2 + lj)
                        U[u_idx][col] += val
                    
                    val = v_loc[li * 2 + lj]
                    if val != 0:
                        v_idx = (bb_r * 2 + li) * 4 + (bb_c * 2 + lj)
                        V[v_idx][col] += val
                    
                    val = w_loc[li * 2 + lj]
                    if val != 0:
                        # W index is k * n + i, where k is col of C, i is row of C?
                        # Task: W[k*n+i, r]. C is p x n? No, C is n x p?
                        # Matrix mult: A (n x m) * B (m x p) = C (n x p).
                        # Here n=m=p=4.
                        # C indices: row i, col k.
                        # Task says W[k*n+i]. So index is k*4 + i.
                        # In my block logic:
                        # C block (cb_r, cb_c).
                        # Global row i = cb_r * 2 + li
                        # Global col k = cb_c * 2 + lj
                        # So W index = k * 4 + i = (cb_c * 2 + lj) * 4 + (cb_r * 2 + li)
                        
                        w_idx = (cb_c * 2 + lj) * 4 + (cb_r * 2 + li)
                        W[w_idx][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
