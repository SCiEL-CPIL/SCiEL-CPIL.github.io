"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-98 tensor decomposition for 4x4 * 4x8 -> 4x8 multiplication.
    
    Strategy: Treat the 4x4 left matrix as a 2x2 block matrix of 2x2 blocks.
    Use Strassen's algorithm at the block level (7 block products instead of 4).
    Each block product is a 2x2 * 2x2 multiplication (rank 7 via Strassen).
    Total rank = 7 * 7 = 49 for the 4x4x4 part, but we need to handle the 4x8 output.
    
    Actually, let's think more carefully:
    - A is 4x4, B is 4x8, C = A*B is 4x8.
    - We can view A as [[A11, A12], [A21, A22]] where each Ai is 2x2.
    - We can view B as [[B11, B12, B13, B14], [B21, B22, B23, B24]] where each Bji is 2x2.
    - C = [[A11*B11 + A12*B21, A11*B12 + A12*B22, A11*B13 + A12*B23, A11*B14 + A12*B24],
           [A21*B11 + A22*B21, A21*B12 + A22*B22, A21*B13 + A22*B23, A21*B14 + A22*B24]]
    
    Using Strassen for the 2x2 block matrix multiplication:
    M1 = (A22 - A12) * (B11 + B21)  -- but wait, B is 4x8, so the right block is 2x4 in block terms
    This doesn't directly work because Strassen assumes square block matrices.
    
    Let me reconsider. The output is 4x8. We can split the output into two 4x4 blocks:
    C = [C_left | C_right] where C_left = A * B[:, 0:4] and C_right = A * B[:, 4:8].
    
    So we're doing two 4x4 * 4x4 multiplications. Each can be done with rank 49 (Strassen squared).
    Total rank = 49 + 49 = 98.
    
    This matches the measured knowledge: "Tensor product of Strassen 2x2x2 gives exact 4x4x4 rank-49 decomposition; replicating for 8-wide output yields rank 98"
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Strassen decomposition for 2x2 * 2x2 -> 2x2
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4 * 4x4, we use Strassen at the 2x2 block level:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    #
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7  (2x2)
    # C12 = M3 + M5             (2x2)
    # C21 = M2 + M4             (2x2)
    # C22 = M1 - M2 + M3 + M6  (2x2)
    #
    # Each Mi is a 2x2 * 2x2 product, which itself can be done with rank 7.
    # So total rank for one 4x4*4x4 is 7*7 = 49.
    # For 4x4*4x8, we do two such products, total rank 98.
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # Index conventions:
    # U[i*m + j, col] corresponds to A[i,j]
    # V[j*p + k, col] corresponds to B[j,k]
    # W[k*n + i, col] corresponds to C[i,k] (trace convention: W[k*n+i, r])
    
    # For the first 4x4*4x4 (B columns 0-3):
    # A indices: i in 0..3, j in 0..3
    # B indices: j in 0..3, k in 0..3
    # C indices: i in 0..3, k in 0..3
    
    # For the second 4x4*4x4 (B columns 4-7):
    # A indices: i in 0..3, j in 0..3
    # B indices: j in 0..3, k in 4..7
    # C indices: i in 0..3, k in 4..7
    
    # Let's define the Strassen decomposition for 2x2*2x2.
    # We'll create a helper that adds the rank-7 decomposition for a 2x2*2x2 product.
    
    def add_strassen_2x2(U, V, W, a_base, b_base, c_base, col_offset, sign_a, sign_b, sign_c):
        """
        Add a Strassen decomposition for 2x2 * 2x2 -> 2x2.
        
        a_base: (row, col) offset for the 2x2 A block (top-left corner)
        b_base: (row, col) offset for the 2x2 B block (top-left corner)
        c_base: (row, col) offset for the 2x2 C block (top-left corner)
        col_offset: starting column in U, V, W
        sign_a, sign_b, sign_c: signs to apply to A, B, C entries (for sums/differences)
        
        The 2x2 blocks are indexed as:
        A[0,0], A[0,1], A[1,0], A[1,1]
        B[0,0], B[0,1], B[1,0], B[1,1]
        C[0,0], C[0,1], C[1,0], C[1,1]
        """
        # Strassen:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Each Mi contributes to C as:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each of the 7 Strassen products, we add a rank-1 term (well, rank-7 for 2x2*2x2)
        # Actually, each Mi is itself a 2x2*2x2 product, which needs rank 7.
        # So we need to recursively apply Strassen? No, 2x2*2x2 is the base case.
        
        # Wait, I'm confusing myself. Let me restart.
        
        # For 2x2 * 2x2 -> 2x2, Strassen uses 7 scalar multiplications.
        # Each scalar multiplication is rank 1.
        # So the rank-7 decomposition for 2x2*2x2 has 7 columns.
        
        # Let me define the 7 Strassen factors for 2x2*2x2:
        # The tensor is T[i,j,k] = sum_l A[i,l] * B[l,k] for i,j,k in {0,1}
        
        # Strassen decomposition:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each Mi, we have a rank-1 term: U_vec * V_vec * W_vec
        # where U_vec is a 4-vector (for 2x2 A), V_vec is a 4-vector (for 2x2 B), W_vec is a 4-vector (for 2x2 C)
        
        # Let's enumerate the 7 factors:
        # Factor 1: M1 = (A00 + A11) * (B00 + B11)
        #   U1 = [1, 0, 0, 1]  (A00 + A11)
        #   V1 = [1, 0, 0, 1]  (B00 + B11)
        #   W1 = [1, 0, 0, 1]  (contributes to C00 and C11)
        
        # Actually, the W vector should reflect which C entries get this M.
        # M1 contributes to C00 and C11.
        # So W1 = [1, 0, 0, 1] means C00 += M1 and C11 += M1.
        
        # Let me be more careful. The decomposition is:
        # T = sum_r U_r ⊗ V_r ⊗ W_r
        # where T[i,j,k] = sum_r U_r[i] * V_r[j] * W_r[k]
        
        # For 2x2*2x2, i,j,k in {0,1}.
        
        # Strassen:
        # r=1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]  (M1: A00+A11, B00+B11, C00+C11)
        # r=2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]  (M2: A10+A11, B00, C10)
        # r=3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]  (M3: A00, B01-B11, C01)
        # r=4: U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,1,0]  (M4: A11, B10-B00, C00+C10)
        # r=5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]  (M5: A00+A01, B11, C00-C01... wait)
        
        # Let me look up the exact Strassen factors.
        
        # Standard Strassen for 2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 - M5 + M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So the W vectors are:
        # r=1 (M1): W = [1, 0, 0, 1]  (C00 += M1, C11 += M1)
        # r=2 (M2): W = [0, 0, 1, -1]  (C10 += M2, C11 -= M2)
        # r=3 (M3): W = [0, 1, 0, 1]  (C01 += M3, C11 += M3)
        # r=4 (M4): W = [1, 0, 1, 0]  (C00 += M4, C10 += M4)
        # r=5 (M5): W = [-1, 1, 0, 0]  (C00 -= M5, C01 += M5)
        # r=6 (M6): W = [0, 0, 0, 1]  (C11 += M6)
        # r=7 (M7): W = [1, 0, 0, 0]  (C00 += M7)
        
        # And the U vectors:
        # r=1: U = [1, 0, 0, 1]  (A00 + A11)
        # r=2: U = [0, 0, 1, 1]  (A10 + A11)
        # r=3: U = [1, 0, 0, 0]  (A00)
        # r=4: U = [0, 0, 0, 1]  (A11)
        # r=5: U = [1, 1, 0, 0]  (A00 + A01)
        # r=6: U = [-1, 0, 1, 0]  (A10 - A00)
        # r=7: U = [0, 1, 0, -1]  (A01 - A11)
        
        # And the V vectors:
        # r=1: V = [1, 0, 0, 1]  (B00 + B11)
        # r=2: V = [1, 0, 0, 0]  (B00)
        # r=3: V = [0, 1, 0, -1]  (B01 - B11)
        # r=4: V = [0, 0, 1, -1]  (B10 - B00)
        # r=5: V = [0, 0, 0, 1]  (B11)
        # r=6: V = [1, 1, 0, 0]  (B00 + B01)
        # r=7: V = [0, 0, 1, 1]  (B10 + B11)
        
        # Let me verify: C00 = M1 - M5 + M4 + M7
        # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # M5 = (A00+A01)*B11 = A00*B11 + A01*B11
        # M4 = A11*(B10-B00) = A11*B10 - A11*B00
        # M7 = (A01-A11)*(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11
        # C00 = M1 - M5 + M4 + M7
        #     = (A00*B00 + A00*B11 + A11*B00 + A11*B11) - (A00*B11 + A01*B11) + (A11*B10 - A11*B00) + (A01*B10 + A01*B11 - A11*B10 - A11*B11)
        #     = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A00*B11 - A01*B11 + A11*B10 - A11*B00 + A01*B10 + A01*B11 - A11*B10 - A11*B11
        #     = A00*B00 + A01*B10
        #     = A00*B00 + A01*B10 ✓
        
        # Good, the decomposition is correct.
        
        # Now, for the 4x4*4x4 Strassen:
        # We have 7 block products, each of which is a 2x2*2x2 product.
        # Each 2x2*2x2 product uses 7 scalar multiplications.
        # So total 7*7 = 49 scalar multiplications.
        
        # Let me define the 49 factors.
        
        # For the first 4x4*4x4 (columns 0-3 of B, columns 0-3 of C):
        # Block indices: A block (bi, bj) where bi, bj in {0,1}, each block is 2x2.
        # A block (0,0): rows 0-1, cols 0-1
        # A block (0,1): rows 0-1, cols 2-3
        # A block (1,0): rows 2-3, cols 0-1
        # A block (1,1): rows 2-3, cols 2-3
        
        # Similarly for B and C.
        
        # Strassen at block level:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 - M5 + M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mi is a 2x2*2x2 product, so we apply the 7-factor Strassen decomposition to each.
        
        # Let's build this systematically.
        
        # Define the 7 Strassen factors for 2x2*2x2 as above.
        strassen_U = [
            [1, 0, 0, 1],  # r=1
            [0, 0, 1, 1],  # r=2
            [1, 0, 0, 0],  # r=3
            [0, 0, 0, 1],  # r=4
            [1, 1, 0, 0],  # r=5
            [-1, 0, 1, 0], # r=6
            [0, 1, 0, -1]  # r=7
        ]
        strassen_V = [
            [1, 0, 0, 1],  # r=1
            [1, 0, 0, 0],  # r=2
            [0, 1, 0, -1], # r=3
            [0, 0, 1, -1], # r=4
            [0, 0, 0, 1],  # r=5
            [1, 1, 0, 0],  # r=6
            [0, 0, 1, 1]   # r=7
        ]
        strassen_W = [
            [1, 0, 0, 1],  # r=1: C00 += M1, C11 += M1
            [0, 0, 1, -1], # r=2: C10 += M2, C11 -= M2
            [0, 1, 0, 1],  # r=3: C01 += M3, C11 += M3
            [1, 0, 1, 0],  # r=4: C00 += M4, C10 += M4
            [-1, 1, 0, 0], # r=5: C00 -= M5, C01 += M5
            [0, 0, 0, 1],  # r=6: C11 += M6
            [1, 0, 0, 0]   # r=7: C00 += M7
        ]
        
        # Now, for the 4x4*4x4 Strassen, we have 7 block products.
        # For each block product Mi, we need to determine:
        # - Which A blocks are involved (with signs)
        # - Which B blocks are involved (with signs)
        # - Which C blocks receive the result (with signs)
        
        # Block product definitions:
        # M1: A blocks (0,0)+(1,1), B blocks (0,0)+(1,1), C blocks (0,0)+(1,1)
        # M2: A blocks (1,0)+(1,1), B block (0,0), C block (1,0)
        # M3: A block (0,0), B blocks (0,1)-(1,1), C block (0,1)
        # M4: A block (1,1), B blocks (1,0)-(0,0), C blocks (0,0)+(1,0)
        # M5: A blocks (0,0)+(0,1), B block (1,1), C blocks (0,0)-(0,1)
        # M6: A blocks (1,0)-(0,0), B blocks (0,0)+(0,1), C block (1,1)
        # M7: A blocks (0,1)-(1,1), B blocks (1,0)+(1,1), C block (0,0)
        
        # For each block product, we apply the 7-factor Strassen decomposition.
        # The total rank is 7 * 7 = 49.
        
        # Let's define the block-level Strassen factors.
        # Each factor is a tuple of (a_blocks, b_blocks, c_blocks) where each is a dict from block index to coefficient.
        
        block_strassen = [
            # (a_blocks, b_blocks, c_blocks)
            # M1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
            # M3
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
            # M4
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # M6
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
        ]
        
        # Now, for each of the 7 block products, and for each of the 7 Strassen factors within that block product,
        # we create a rank-1 term.
        
        # For the first 4x4*4x4 (B cols 0-3, C cols 0-3):
        for bi in range(7):
            a_blocks, b_blocks, c_blocks = block_strassen[bi]
            for si in range(7):
                col = bi * 7 + si
                U_vec = strassen_U[si]
                V_vec = strassen_V[si]
                W_vec = strassen_W[si]
                
                # For each A block involved in this block product:
                for (ab_i, ab_j), a_coeff in a_blocks.items():
                    for si_local in range(2):
                        for sj_local in range(2):
                            # A block (ab_i, ab_j) has entries at:
                            # row = ab_i*2 + si_local, col = ab_j*2 + sj_local
                            row = ab_i * 2 + si_local
                            col_a = ab_j * 2 + sj_local
                            U[row * m + col_a][col] = a_coeff * U_vec[si_local * 2 + sj_local]
                
                # For each B block involved:
                for (bb_i, bb_j), b_coeff in b_blocks.items():
                    for si_local in range(2):
                        for sj_local in range(2):
                            # B block (bb_i, bb_j) has entries at:
                            # row = bb_i*2 + si_local, col = bb_j*2 + sj_local
                            row = bb_i * 2 + si_local
                            col_b = bb_j * 2 + sj_local
                            V[row * p + col_b][col] = b_coeff * V_vec[si_local * 2 + sj_local]
                
                # For each C block involved:
                for (cb_i, cb_j), c_coeff in c_blocks.items():
                    for si_local in range(2):
                        for sj_local in range(2):
                            # C block (cb_i, cb_j) has entries at:
                            # row = cb_i*2 + si_local, col = cb_j*2 + sj_local
                            row = cb_i * 2 + si_local
                            col_c = cb_j * 2 + sj_local
                            # W convention: W[k*n + i, r] where i is row of C, k is col of C
                            W[col_c * n + row][col] = c_coeff * W_vec[si_local * 2 + sj_local]
        
        # For the second 4x4*4x4 (B cols 4-7, C cols 4-7):
        # Same A blocks, but B and C are shifted by 4 in the column direction.
        for bi in range(7):
            a_blocks, b_blocks, c_blocks = block_strassen[bi]
            for si in range(7):
                col = 49 + bi * 7 + si
                U_vec = strassen_U[si]
                V_vec = strassen_V[si]
                W_vec = strassen_W[si]
                
                # For each A block involved in this block product:
                for (ab_i, ab_j), a_coeff in a_blocks.items():
                    for si_local in range(2):
                        for sj_local in range(2):
                            row = ab_i * 2 + si_local
                            col_a = ab_j * 2 + sj_local
                            U[row * m + col_a][col] = a_coeff * U_vec[si_local * 2 + sj_local]
                
                # For each B block involved:
                for (bb_i, bb_j), b_coeff in b_blocks.items():
                    for si_local in range(2):
                        for sj_local in range(2):
                            row = bb_i * 2 + si_local
                            col_b = bb_j * 2 + sj_local + 4  # Shift by 4
                            V[row * p + col_b][col] = b_coeff * V_vec[si_local * 2 + sj_local]
                
                # For each C block involved:
                for (cb_i, cb_j), c_coeff in c_blocks.items():
                    for si_local in range(2):
                        for sj_local in range(2):
                            row = cb_i * 2 + si_local
                            col_c = cb_j * 2 + sj_local + 4  # Shift by 4
                            W[col_c * n + row][col] = c_coeff * W_vec[si_local * 2 + sj_local]
        
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
