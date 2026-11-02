"""Program for matrix multiplication 3x4x4 tensor decomposition.
Implements a block decomposition using Strassen's algorithm for 2x2 blocks
and schoolbook multiplication for 1x2 blocks, achieving rank 44.
"""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We'll build the decomposition using a block approach:
    # Split A (3x4) into A_top (1x4) and A_bot (2x4)
    # Split B (4x4) into 2x2 blocks: B11, B12, B21, B22
    # 
    # For A_bot (2x4) * B (4x4):
    #   Split into A_b1 (2x2) * B11, A_b1 (2x2) * B12, A_b2 (2x2) * B21, A_b2 (2x2) * B22
    #   Each 2x2*2x2 uses Strassen (rank 7), so 4*7 = 28 terms
    #
    # For A_top (1x4) * B (4x4):
    #   Split into A_t1 (1x2) * B11, A_t1 (1x2) * B12, A_t2 (1x2) * B21, A_t2 (1x2) * B22
    #   Each 1x2*2x2 uses schoolbook (rank 4), so 4*4 = 16 terms
    #
    # Total rank: 28 + 16 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1
    
    # Indexing:
    # U[i*m + j] for A[i,j], i in 0..2, j in 0..3
    # V[j*p + k] for B[j,k], j in 0..3, k in 0..3
    # W[k*n + i] for C[k,i], k in 0..3, i in 0..2
    
    # Strassen 2x2 multiplication
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X * Y
    # m0 = (x00+x11)(y00+y11)
    # m1 = (x10+x11)y00
    # m2 = x00(y01-y11)
    # m3 = x11(y10-y00)
    # m4 = (x00+x01)y11
    # m5 = (x10-x00)(y00+y01)
    # m6 = (x01-x11)(y10+y11)
    # z00 = m0 - m3 + m5 - m6
    # z01 = m2 + m4
    # z10 = m1 + m3
    # z11 = m0 - m2 + m1 - m4
    
    def strassen_block(x_indices, y_indices, z_indices):
        """
        x_indices: (x00, x01, x10, x11) - U indices
        y_indices: (y00, y01, y10, y11) - V indices
        z_indices: (z00, z01, z10, z11) - W indices
        """
        x00, x01, x10, x11 = x_indices
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # m0 = (x00+x11)(y00+y11)
        add_term(x00, y00, z00, 1, 1, 1)
        add_term(x00, y11, z00, 1, 1, 1)
        add_term(x11, y00, z00, 1, 1, 1)
        add_term(x11, y11, z00, 1, 1, 1)
        
        # m1 = (x10+x11)y00
        add_term(x10, y00, z10, 1, 1, 1)
        add_term(x11, y00, z10, 1, 1, 1)
        
        # m2 = x00(y01-y11)
        add_term(x00, y01, z01, 1, 1, 1)
        add_term(x00, y11, z01, 1, -1, 1)
        
        # m3 = x11(y10-y00)
        add_term(x11, y10, z10, 1, 1, 1)
        add_term(x11, y00, z10, 1, -1, 1)
        
        # m4 = (x00+x01)y11
        add_term(x00, y11, z01, 1, 1, 1)
        add_term(x01, y11, z01, 1, 1, 1)
        
        # m5 = (x10-x00)(y00+y01)
        add_term(x10, y00, z00, 1, 1, 1)
        add_term(x10, y01, z00, 1, 1, 1)
        add_term(x00, y00, z00, -1, 1, 1)
        add_term(x00, y01, z00, -1, 1, 1)
        
        # m6 = (x01-x11)(y10+y11)
        add_term(x01, y10, z00, 1, 1, 1)
        add_term(x01, y11, z00, 1, 1, 1)
        add_term(x11, y10, z00, -1, 1, 1)
        add_term(x11, y11, z00, -1, 1, 1)
        
        # Now apply the output combinations
        # z00 = m0 - m3 + m5 - m6
        # z01 = m2 + m4
        # z10 = m1 + m3
        # z11 = m0 - m2 + m1 - m4
        
        # We need to track which terms go to which output
        # This is getting complex. Let me use a different approach.
        pass
    
    # Let me use a cleaner approach: directly add the 7 Strassen terms
    # with proper output coefficients
    
    def strassen_2x2(x_idx, y_idx, z_idx):
        """
        x_idx: list of 4 U indices [x00, x01, x10, x11]
        y_idx: list of 4 V indices [y00, y01, y10, y11]
        z_idx: list of 4 W indices [z00, z01, z10, z11]
        """
        x00, x01, x10, x11 = x_idx
        y00, y01, y10, y11 = y_idx
        z00, z01, z10, z11 = z_idx
        
        # m0 = (x00+x11)(y00+y11)
        # Contribution: z00 += 1, z11 += 1
        add_term(x00, y00, z00, 1, 1, 1)
        add_term(x00, y11, z00, 1, 1, 1)
        add_term(x11, y00, z00, 1, 1, 1)
        add_term(x11, y11, z00, 1, 1, 1)
        add_term(x00, y00, z11, 1, 1, 1)
        add_term(x00, y11, z11, 1, 1, 1)
        add_term(x11, y00, z11, 1, 1, 1)
        add_term(x11, y11, z11, 1, 1, 1)
        
        # m1 = (x10+x11)y00
        # Contribution: z10 += 1, z11 += 1
        add_term(x10, y00, z10, 1, 1, 1)
        add_term(x11, y00, z10, 1, 1, 1)
        add_term(x10, y00, z11, 1, 1, 1)
        add_term(x11, y00, z11, 1, 1, 1)
        
        # m2 = x00(y01-y11)
        # Contribution: z01 += 1, z11 += -1
        add_term(x00, y01, z01, 1, 1, 1)
        add_term(x00, y11, z01, 1, -1, 1)
        add_term(x00, y01, z11, 1, 1, -1)
        add_term(x00, y11, z11, 1, -1, -1)
        
        # m3 = x11(y10-y00)
        # Contribution: z10 += 1, z00 += -1
        add_term(x11, y10, z10, 1, 1, 1)
        add_term(x11, y00, z10, 1, -1, 1)
        add_term(x11, y10, z00, 1, 1, -1)
        add_term(x11, y00, z00, 1, -1, -1)
        
        # m4 = (x00+x01)y11
        # Contribution: z01 += 1, z11 += -1
        add_term(x00, y11, z01, 1, 1, 1)
        add_term(x01, y11, z01, 1, 1, 1)
        add_term(x00, y11, z11, 1, 1, -1)
        add_term(x01, y11, z11, 1, 1, -1)
        
        # m5 = (x10-x00)(y00+y01)
        # Contribution: z00 += 1
        add_term(x10, y00, z00, 1, 1, 1)
        add_term(x10, y01, z00, 1, 1, 1)
        add_term(x00, y00, z00, -1, 1, 1)
        add_term(x00, y01, z00, -1, 1, 1)
        
        # m6 = (x01-x11)(y10+y11)
        # Contribution: z00 += -1
        add_term(x01, y10, z00, 1, 1, -1)
        add_term(x01, y11, z00, 1, 1, -1)
        add_term(x11, y10, z00, -1, 1, -1)
        add_term(x11, y11, z00, -1, 1, -1)
    
    # This approach is adding too many terms. Let me reconsider.
    # The issue is that each Strassen term m_i is a product of a sum in U and a sum in V.
    # We need to represent this as a single term in the decomposition, but with coefficients.
    
    # Actually, the correct way is:
    # m0 = (x00+x11)(y00+y11) means we have a term where:
    #   U has +1 at x00 and +1 at x11
    #   V has +1 at y00 and +1 at y11
    #   W has coefficient 1 at z00 and coefficient 1 at z11 (since m0 contributes to both)
    
    # But in a rank-1 term, we can only have one U vector, one V vector, one W vector.
    # So m0 = (x00+x11)(y00+y11) requires 4 rank-1 terms if we expand.
    
    # Wait, that's not right either. In tensor decomposition:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # For m0 to contribute to z00 and z11, we need:
    #   One term: u = e_x00 + e_x11, v = e_y00 + e_y11, w = e_z00 + e_z11
    # This is a single rank-1 term!
    
    # So let me redo this properly.
    
    # Reset
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
        """
        Add a rank-1 term.
        u_indices: list of U indices
        v_indices: list of V indices  
        w_indices: list of W indices
        u_coeffs: list of coefficients for U
        v_coeffs: list of coefficients for V
        w_coeffs: list of coefficients for W
        """
        nonlocal col
        if col >= rank:
            return
        for idx, coeff in zip(u_indices, u_coeffs):
            U[idx][col] += coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            V[idx][col] += coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            W[idx][col] += coeff
        col += 1
    
    # Strassen 2x2 block
    def strassen_2x2_block(x_idx, y_idx, z_idx):
        """
        x_idx: [x00, x01, x10, x11]
        y_idx: [y00, y01, y10, y11]
        z_idx: [z00, z01, z10, z11]
        """
        x00, x01, x10, x11 = x_idx
        y00, y01, y10, y11 = y_idx
        z00, z01, z10, z11 = z_idx
        
        # m0 = (x00+x11)(y00+y11)
        # z00 += m0, z11 += m0
        add_term([x00, x11], [y00, y11], [z00, z11], [1, 1], [1, 1], [1, 1])
        
        # m1 = (x10+x11)y00
        # z10 += m1, z11 += m1
        add_term([x10, x11], [y00], [z10, z11], [1, 1], [1], [1, 1])
        
        # m2 = x00(y01-y11)
        # z01 += m2, z11 += -m2
        add_term([x00], [y01, y11], [z01, z11], [1], [1, -1], [1, -1])
        
        # m3 = x11(y10-y00)
        # z10 += m3, z00 += -m3
        add_term([x11], [y10, y00], [z10, z00], [1], [1, -1], [1, -1])
        
        # m4 = (x00+x01)y11
        # z01 += m4, z11 += -m4
        add_term([x00, x01], [y11], [z01, z11], [1, 1], [1], [1, -1])
        
        # m5 = (x10-x00)(y00+y01)
        # z00 += m5
        add_term([x10, x00], [y00, y01], [z00], [1, -1], [1, 1], [1])
        
        # m6 = (x01-x11)(y10+y11)
        # z00 += -m6
        add_term([x01, x11], [y10, y11], [z00], [1, -1], [1, 1], [-1])
    
    # Schoolbook 1x2 * 2x2 block
    def schoolbook_1x2_block(x_idx, y_idx, z_idx):
        """
        x_idx: [x0, x1] (1x2 matrix)
        y_idx: [y00, y01, y10, y11] (2x2 matrix)
        z_idx: [z0, z1] (1x2 output)
        z0 = x0*y00 + x1*y10
        z1 = x0*y01 + x1*y11
        """
        x0, x1 = x_idx
        y00, y01, y10, y11 = y_idx
        z0, z1 = z_idx
        
        # z0 += x0*y00
        add_term([x0], [y00], [z0], [1], [1], [1])
        # z0 += x1*y10
        add_term([x1], [y10], [z0], [1], [1], [1])
        # z1 += x0*y01
        add_term([x0], [y01], [z1], [1], [1], [1])
        # z1 += x1*y11
        add_term([x1], [y11], [z1], [1], [1], [1])
    
    # Now apply to our problem
    # A is 3x4, B is 4x4, C is 4x3 (output convention C[k,i])
    # i in 0..2 (rows of A), j in 0..3 (cols of A / rows of B), k in 0..3 (cols of B / rows of C)
    
    # Split A:
    # A_top = A[0, :] (1x4): indices U[0*4+0], U[0*4+1], U[0*4+2], U[0*4+3]
    # A_bot = A[1:3, :] (2x4): 
    #   A_b1 = A[1:3, 0:2] (2x2): U[1*4+0], U[1*4+1], U[2*4+0], U[2*4+1]
    #   A_b2 = A[1:3, 2:4] (2x2): U[1*4+2], U[1*4+3], U[2*4+2], U[2*4+3]
    
    # Split B:
    # B11 = B[0:2, 0:2]: V[0*4+0], V[0*4+1], V[1*4+0], V[1*4+1]
    # B12 = B[0:2, 2:4]: V[0*4+2], V[0*4+3], V[1*4+2], V[1*4+3]
    # B21 = B[2:4, 0:2]: V[2*4+0], V[2*4+1], V[3*4+0], V[3*4+1]
    # B22 = B[2:4, 2:4]: V[2*4+2], V[2*4+3], V[3*4+2], V[3*4+3]
    
    # Output C (4x3), C[k,i]:
    # For A_bot * B11: contributes to C[0:2, 1:3]
    #   z00 -> C[0,1] = W[0*3+1] = W[1]
    #   z01 -> C[0,2] = W[0*3+2] = W[2]
    #   z10 -> C[1,1] = W[1*3+1] = W[4]
    #   z11 -> C[1,2] = W[1*3+2] = W[5]
    
    # For A_bot * B12: contributes to C[2:4, 1:3]
    #   z00 -> C[2,1] = W[2*3+1] = W[7]
    #   z01 -> C[2,2] = W[2*3+2] = W[8]
    #   z10 -> C[3,1] = W[3*3+1] = W[10]
    #   z11 -> C[3,2] = W[3*3+2] = W[11]
    
    # For A_bot * B21: contributes to C[0:2, 1:3]
    #   Same as B11 output
    
    # For A_bot * B22: contributes to C[2:4, 1:3]
    #   Same as B12 output
    
    # For A_top * B11: contributes to C[0:2, 0]
    #   z0 -> C[0,0] = W[0*3+0] = W[0]
    #   z1 -> C[1,0] = W[1*3+0] = W[3]
    
    # For A_top * B12: contributes to C[2:4, 0]
    #   z0 -> C[2,0] = W[2*3+0] = W[6]
    #   z1 -> C[3,0] = W[3*3+0] = W[9]
    
    # For A_top * B21: contributes to C[0:2, 0]
    # For A_top * B22: contributes to C[2:4, 0]
    
    # Apply Strassen for A_bot * B blocks
    # A_b1 * B11
    strassen_2x2_block(
        [1*4+0, 1*4+1, 2*4+0, 2*4+1],  # x00, x01, x10, x11
        [0*4+0, 0*4+1, 1*4+0, 1*4+1],  # y00, y01, y10, y11
        [1, 2, 4, 5]                     # z00, z01, z10, z11
    )
    
    # A_b1 * B12
    strassen_2x2_block(
        [1*4+0, 1*4+1, 2*4+0, 2*4+1],
        [0*4+2, 0*4+3, 1*4+2, 1*4+3],
        [7, 8, 10, 11]
    )
    
    # A_b2 * B21
    strassen_2x2_block(
        [1*4+2, 1*4+3, 2*4+2, 2*4+3],
        [2*4+0, 2*4+1, 3*4+0, 3*4+1],
        [1, 2, 4, 5]
    )
    
    # A_b2 * B22
    strassen_2x2_block(
        [1*4+2, 1*4+3, 2*4+2, 2*4+3],
        [2*4+2, 2*4+3, 3*4+2, 3*4+3],
        [7, 8, 10, 11]
    )
    
    # Apply schoolbook for A_top * B blocks
    # A_top = [U[0], U[1], U[2], U[3]]
    # A_t1 = [U[0], U[1]] (cols 0,1)
    # A_t2 = [U[2], U[3]] (cols 2,3)
    
    # A_t1 * B11
    schoolbook_1x2_block(
        [0*4+0, 0*4+1],  # x0, x1
        [0*4+0, 0*4+1, 1*4+0, 1*4+1],  # y00, y01, y10, y11
        [0, 3]  # z0, z1
    )
    
    # A_t1 * B12
    schoolbook_1x2_block(
        [0*4+0, 0*4+1],
        [0*4+2, 0*4+3, 1*4+2, 1*4+3],
        [6, 9]
    )
    
    # A_t2 * B21
    schoolbook_1x2_block(
        [0*4+2, 0*4+3],
        [2*4+0, 2*4+1, 3*4+0, 3*4+1],
        [0, 3]
    )
    
    # A_t2 * B22
    schoolbook_1x2_block(
        [0*4+2, 0*4+3],
        [2*4+2, 2*4+3, 3*4+2, 3*4+3],
        [6, 9]
    )
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
