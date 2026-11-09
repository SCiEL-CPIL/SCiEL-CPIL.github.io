"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We use a block decomposition approach.
    # Split the 4x4x6 problem into:
    #   Part 1: 4x4x4 (first 4 columns of B) - 8 blocks of 2x2x2
    #   Part 2: 4x4x2 (last 2 columns of B) - 4 blocks of 2x2x2
    #
    # For each 2x2x2 block, we use Strassen's algorithm with rank 7.
    # 
    # Strassen's algorithm for 2x2 matrices:
    # Let A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A * B = [[C00, C01], [C10, C11]]
    #
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # To express this as a tensor decomposition, we need to write each Mt
    # as a sum of rank-1 terms. Since each Mt is a scalar product of two
    # linear combinations, we can write:
    # Mt = (sum_a ua_a * A_a) * (sum_b vb_b * B_b)
    # This is a rank-1 term in the tensor sense only if we consider the
    # full tensor. But actually, each Mt contributes to multiple C entries.
    #
    # The correct way: The 2x2x2 multiplication tensor has rank 7.
    # The 7 Strassen terms give us 7 rank-1 decompositions of the tensor.
    # Each rank-1 term is u_t ⊗ v_t ⊗ w_t where:
    #   u_t is a vector in R^4 (coefficients for A00,A01,A10,A11)
    #   v_t is a vector in R^4 (coefficients for B00,B01,B10,B11)
    #   w_t is a vector in R^4 (coefficients for C00,C01,C10,C11)
    #
    # Let me derive the explicit factors:
    #
    # For M1 = (A00+A11)(B00+B11):
    #   This contributes to C00 and C11.
    #   C00 gets +M1, C11 gets +M1.
    #   u1 = [1, 0, 0, 1]  (A00 + A11)
    #   v1 = [1, 0, 0, 1]  (B00 + B11)
    #   w1 = [1, 0, 0, 1]  (C00 + C11)
    #
    # For M2 = (A10+A11) * B00:
    #   C10 gets +M2, C11 gets -M2.
    #   u2 = [0, 0, 1, 1]  (A10 + A11)
    #   v2 = [1, 0, 0, 0]  (B00)
    #   w2 = [0, 0, 1, -1] (C10 - C11)
    #
    # For M3 = A00 * (B01-B11):
    #   C01 gets +M3, C11 gets +M3.
    #   u3 = [1, 0, 0, 0]  (A00)
    #   v3 = [0, 1, 0, -1] (B01 - B11)
    #   w3 = [0, 1, 0, 1]  (C01 + C11)
    #
    # For M4 = A11 * (B10-B00):
    #   C00 gets +M4, C10 gets +M4.
    #   u4 = [0, 0, 0, 1]  (A11)
    #   v4 = [0, 0, 1, -1] (B10 - B00)
    #   w4 = [1, 0, 1, 0]  (C00 + C10)
    #
    # For M5 = (A00+A01) * B11:
    #   C00 gets -M5, C01 gets +M5.
    #   u5 = [1, 1, 0, 0]  (A00 + A01)
    #   v5 = [0, 0, 0, 1]  (B11)
    #   w5 = [-1, 1, 0, 0] (-C00 + C01)
    #
    # For M6 = (A10-A00) * (B00+B01):
    #   C11 gets +M6.
    #   u6 = [-1, 0, 1, 0] (-A00 + A10)
    #   v6 = [1, 1, 0, 0]  (B00 + B01)
    #   w6 = [0, 0, 0, 1]  (C11)
    #
    # For M7 = (A01-A11) * (B10+B11):
    #   C00 gets +M7.
    #   u7 = [0, 1, 0, -1] (A01 - A11)
    #   v7 = [0, 0, 1, 1]  (B10 + B11)
    #   w7 = [1, 0, 0, 0]  (C00)
    
    # Let me verify: 
    # C00 = M1 + M4 - M5 + M7
    #   = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    #   Let me expand:
    #   M1 = A00*B00 + A00*B11 + A11*B00 + A11*B11
    #   M4 = A11*B10 - A11*B00
    #   M5 = A00*B11 + A01*B11
    #   M7 = A01*B10 + A01*B11 - A11*B10 - A11*B11
    #   C00 = M1 + M4 - M5 + M7
    #       = (A00*B00 + A00*B11 + A11*B00 + A11*B11) 
    #         + (A11*B10 - A11*B00) 
    #         - (A00*B11 + A01*B11) 
    #         + (A01*B10 + A01*B11 - A11*B10 - A11*B11)
    #       = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    #       = A00*B00 + A01*B10
    #   Correct! C00 = A00*B00 + A01*B10.
    
    # So the 7 factors for 2x2x2 are:
    # u1 = [1, 0, 0, 1], v1 = [1, 0, 0, 1], w1 = [1, 0, 0, 1]
    # u2 = [0, 0, 1, 1], v2 = [1, 0, 0, 0], w2 = [0, 0, 1, -1]
    # u3 = [1, 0, 0, 0], v3 = [0, 1, 0, -1], w3 = [0, 1, 0, 1]
    # u4 = [0, 0, 0, 1], v4 = [0, 0, 1, -1], w4 = [1, 0, 1, 0]
    # u5 = [1, 1, 0, 0], v5 = [0, 0, 0, 1], w5 = [-1, 1, 0, 0]
    # u6 = [-1, 0, 1, 0], v6 = [1, 1, 0, 0], w6 = [0, 0, 0, 1]
    # u7 = [0, 1, 0, -1], v7 = [0, 0, 1, 1], w7 = [1, 0, 0, 0]
    
    # Now for the full 4x4x6 problem:
    # We split into 2x2 blocks.
    # A is 4x4, B is 4x6, C is 4x6.
    # 
    # A blocks: A[r,c] for r,c in {0,1}, each 2x2
    # B blocks: B[k,c] for k in {0,1}, c in {0,1,2}, each 2x2
    # C blocks: C[r,c] for r in {0,1}, c in {0,1,2}, each 2x2
    #
    # For each output block C[r,c], we need:
    # C[r,c] = A[r,0] * B[0,c] + A[r,1] * B[1,c]
    #
    # Each 2x2x2 product uses 7 rank-1 terms.
    # Total blocks: 2 * 3 = 6 output blocks, each with 2 input block products.
    # Total 2x2x2 products: 6 * 2 = 12.
    # Total rank: 12 * 7 = 84.
    
    # Let me build this.
    # U has shape (n*m, rank) = (16, 84)
    # V has shape (m*p, rank) = (24, 84)  
    # W has shape (p*n, rank) = (24, 84)
    
    # Indexing convention:
    # U[i*m + j] where i is row of A (0..3), j is col of A (0..3)
    # V[j*p + k] where j is row of B (0..3), k is col of B (0..5)
    # W[k*n + i] where k is col of C (0..5), i is row of C (0..3)
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    # Strassen factors for 2x2x2
    # Each factor is (u, v, w) where u,v,w are length-4 vectors
    # u corresponds to A00, A01, A10, A11
    # v corresponds to B00, B01, B10, B11
    # w corresponds to C00, C01, C10, C11
    
    strassen_factors = [
        # (u, v, w)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    
    col = 0
    # For each output block (r, c) where r in {0,1}, c in {0,1,2}
    for r_block in range(2):
        for c_block in range(3):
            # C[r_block, c_block] = A[r_block, 0] * B[0, c_block] + A[r_block, 1] * B[1, c_block]
            for k_block in range(2):
                # This is a 2x2x2 product: A_block * B_block -> C_block
                # A block is at rows r_block*2..r_block*2+1, cols k_block*2..k_block*2+1
                # B block is at rows k_block*2..k_block*2+1, cols c_block*2..c_block*2+1
                # C block is at rows r_block*2..r_block*2+1, cols c_block*2..c_block*2+1
                
                for t in range(7):
                    u, v, w = strassen_factors[t]
                    
                    # u has 4 entries: u[0]=A00, u[1]=A01, u[2]=A10, u[3]=A11
                    # For A block, A00 is at global index (r_block*2 + 0)*m + (k_block*2 + 0)
                    # A01 is at (r_block*2 + 0)*m + (k_block*2 + 1)
                    # A10 is at (r_block*2 + 1)*m + (k_block*2 + 0)
                    # A11 is at (r_block*2 + 1)*m + (k_block*2 + 1)
                    
                    # v has 4 entries: v[0]=B00, v[1]=B01, v[2]=B10, v[3]=B11
                    # For B block, B00 is at global index (k_block*2 + 0)*p + (c_block*2 + 0)
                    # B01 is at (k_block*2 + 0)*p + (c_block*2 + 1)
                    # B10 is at (k_block*2 + 1)*p + (c_block*2 + 0)
                    # B11 is at (k_block*2 + 1)*p + (c_block*2 + 1)
                    
                    # w has 4 entries: w[0]=C00, w[1]=C01, w[2]=C10, w[3]=C11
                    # For C block, C00 is at global index (c_block*2 + 0)*n + (r_block*2 + 0)
                    # C01 is at (c_block*2 + 1)*n + (r_block*2 + 0)
                    # C10 is at (c_block*2 + 0)*n + (r_block*2 + 1)
                    # C11 is at (c_block*2 + 1)*n + (r_block*2 + 1)
                    
                    # Add contributions to U
                    for a_local in range(4):
                        if u[a_local] != 0:
                            # Determine global A index
                            a_row = r_block * 2 + (a_local // 2)  # 0 or 1
                            a_col = k_block * 2 + (a_local % 2)   # 0 or 1
                            a_global = a_row * m + a_col
                            U[a_global][col] += u[a_local]
                    
                    # Add contributions to V
                    for b_local in range(4):
                        if v[b_local] != 0:
                            # Determine global B index
                            b_row = k_block * 2 + (b_local // 2)  # 0 or 1
                            b_col = c_block * 2 + (b_local % 2)   # 0 or 1
                            b_global = b_row * p + b_col
                            V[b_global][col] += v[b_local]
                    
                    # Add contributions to W
                    for c_local in range(4):
                        if w[c_local] != 0:
                            # Determine global C index
                            c_col = c_block * 2 + (c_local % 2)   # 0 or 1
                            c_row = r_block * 2 + (c_local // 2)  # 0 or 1
                            c_global = c_col * n + c_row
                            W[c_global][col] += w[c_local]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
