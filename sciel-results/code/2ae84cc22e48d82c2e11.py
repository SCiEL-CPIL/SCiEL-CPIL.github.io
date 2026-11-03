"""Strassen-based rank-98 decomposition for 4x4 * 4x8 matrix multiplication."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Use rank-7 Strassen decomposition for 4x4 * 4x4, applied to two halves
    of the 4x8 matrix. Total rank = 7 * 7 * 2 = 98.
    
    The 4x4 * 4x4 Strassen decomposition (rank 7) is built from 2x2 Strassen
    with appropriate signs and offsets.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    rank = 98  # 7 (Strassen for 4x4) * 7 (Strassen for 4x4) * 2 (two halves)
    
    # For a 4x4 * 4x4 multiplication, the tensor T[i,j,k] = sum_l A[i,l]*B[l,k]
    # can be decomposed using Strassen's algorithm.
    # 
    # Standard Strassen for 2x2:
    #   M1 = (A11+A22)*(B11+B22)
    #   M2 = (A21+A22)*B11
    #   M3 = A11*(B21-B22)
    #   M4 = A22*(B12-B32)  -- wait, let me use standard notation
    #   M4 = A22*(B12-B22)
    #   M5 = (A11+A12)*B22
    #   M6 = (A21-A11)*(B11+B12)
    #   M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For 4x4, we split into four 2x2 blocks and use a rank-49 construction
    # by applying Strassen at each level. However, the known rank-7 for 4x4
    # is actually not standard -- Strassen gives rank 7 for 2x2, and for 4x4
    # the standard Strassen-based approach gives rank 49 (7^2).
    #
    # Wait -- the prompt says "rank-7 Strassen decomposition along the 4-dimensional
    # output width (n) and the 4-dimensional input height (m)". This is confusing.
    # Let me re-read: "applying the rank-7 Strassen decomposition along the 4-dimensional
    # output width (n) and the 4-dimensional input height (m)"
    #
    # I think the intended approach is:
    # - The 4x4 * 4x4 multiplication has a decomposition of rank 49 using Strassen
    #   (since 7^2 = 49 when you apply 2x2 Strassen to each of the 4 quadrant multiplications)
    # - But actually, there might be a more direct rank-7 interpretation
    #
    # Let me just implement the standard Strassen-based rank-49 for 4x4*4x4,
    # then use it for the two 4x4 halves of the 4x8 problem, giving rank 49*2 = 98.
    
    # Define the 2x2 Strassen decomposition
    # For 2x2 * 2x2: A is 2x2, B is 2x2, C = A*B is 2x2
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Build rank-49 decomposition for 4x4 * 4x4
    # Split 4x4 into four 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = A*B
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each 2x2*2x2 uses rank-7 Strassen, so total rank = 4 * 7 = 28? No, that's not right either.
    # Actually, for 4x4*4x4, the straightforward block approach with 2x2 Strassen gives:
    # 4 quadrant products, each rank 7, total rank 28. But we can do better with
    # a full Strassen recursion, giving 7^2 = 49.
    
    # Let me use the full Strassen recursion for 4x4:
    # Apply Strassen to the 4x4 as if it were a 2x2 of 2x2 blocks.
    # This gives 7 recursive calls, each on 2x2 blocks (rank 7 each).
    # Total rank = 7 * 7 = 49.
    
    # I'll construct this explicitly.
    
    # First, define the 2x2 Strassen factors
    # For 2x2*2x2, U has 4 rows (A entries), V has 4 rows (B entries), W has 4 rows (C entries)
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # Strassen 7 products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b01-b11)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # C entries:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # U factors for 2x2 (row index = a_ij, col index = product m):
    # M1: a00, a11 -> U[0,0]=1, U[3,0]=1
    # M2: a10, a11 -> U[2,1]=1, U[3,1]=1
    # M3: a00 -> U[0,2]=1
    # M4: a11 -> U[3,3]=1
    # M5: a00, a01 -> U[0,4]=1, U[1,4]=1
    # M6: a10, a00 -> U[2,5]=1, U[0,5]=-1
    # M7: a01, a11 -> U[1,6]=1, U[3,6]=-1
    
    U_2x2 = [
        [1, 0, 1, 0, 1, -1, 0],   # a00
        [0, 0, 0, 0, 1, 0, 1],    # a01
        [0, 1, 0, 0, 0, 1, 0],    # a10
        [1, 1, 0, 1, 0, 0, -1]    # a11
    ]
    
    # V factors for 2x2 (row index = b_ij, col index = product m):
    # M1: b00, b11 -> V[0,0]=1, V[3,0]=1
    # M2: b00 -> V[0,1]=1
    # M3: b10, b11 -> V[2,2]=1, V[3,2]=-1
    # M4: b01, b11 -> V[1,3]=1, V[3,3]=-1
    # M5: b11 -> V[3,4]=1
    # M6: b00, b01 -> V[0,5]=1, V[1,5]=1
    # M7: b10, b11 -> V[2,6]=1, V[3,6]=1
    
    V_2x2 = [
        [1, 1, 0, 0, 0, 1, 0],    # b00
        [0, 0, 0, 1, 0, 1, 0],    # b01
        [0, 0, 1, 0, 0, 0, 1],    # b10
        [1, 0, -1, -1, 1, 0, 1]   # b11
    ]
    
    # W factors for 2x2 (row index = c_ij, col index = product m):
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    W_2x2 = [
        [1, 0, 0, 1, -1, 0, 1],   # c00
        [0, 0, 1, 0, 1, 0, 0],    # c01
        [0, 1, 0, 1, 0, 0, 0],    # c10
        [1, -1, 1, 0, 0, 1, 0]    # c11
    ]
    
    # Now build 4x4 * 4x4 rank-49 using Strassen recursion
    # 4x4 A is split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # A00 = rows 0-1, cols 0-1
    # A01 = rows 0-1, cols 2-3
    # A10 = rows 2-3, cols 0-1
    # A11 = rows 2-3, cols 2-3
    
    # Similarly for B.
    
    # Strassen for 4x4 (treating it as 2x2 of 2x2 blocks):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # Each Mi is a 2x2 * 2x2 product, computed with rank-7 Strassen.
    # So total rank = 7 * 7 = 49.
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Cij is 2x2, so C has 4*4=16 entries total.
    
    # U_4x4 has 16 rows (for A entries), 49 columns (for the 49 products)
    # V_4x4 has 16 rows (for B entries), 49 columns
    # W_4x4 has 16 rows (for C entries), 49 columns
    
    U_4x4 = [[0] * 49 for _ in range(16)]
    V_4x4 = [[0] * 49 for _ in range(16)]
    W_4x4 = [[0] * 49 for _ in range(16)]
    
    # The 7 Strassen products, each using 7 sub-products (from 2x2 Strassen)
    # For product q (0-6), sub-product s (0-6), global index = q*7 + s
    
    # A block indices:
    # A00: rows 0,1; cols 0,1 -> A indices 0,1,4,5 (row*4+col)
    # A01: rows 0,1; cols 2,3 -> A indices 2,3,6,7
    # A10: rows 2,3; cols 0,1 -> A indices 8,9,12,13
    # A11: rows 2,3; cols 2,3 -> A indices 10,11,14,15
    
    # Similarly for B and C.
    
    # For each Strassen product q, we need to compute a 2x2*2x2 product.
    # The A-block for product q is determined by the Strassen formula.
    # Similarly for B-block.
    
    # Let me define the A and B block selections for each product q:
    # q=0: A = A00+A11, B = B00+B11
    # q=1: A = A10+A11, B = B00
    # q=2: A = A00, B = B10-B11
    # q=3: A = A11, B = B01-B11
    # q=4: A = A00+A01, B = B11
    # q=5: A = A10-A00, B = B00+B01
    # q=6: A = A01-A11, B = B10+B11
    
    # For each, we apply the 2x2 Strassen with appropriate signs.
    
    # Define block offsets (top-left corner in 4x4 matrix, 0-indexed):
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    
    def a_block_offset(block):
        """Return (row_offset, col_offset) for A block"""
        if block == 0:  # A00
            return (0, 0)
        elif block == 1:  # A01
            return (0, 2)
        elif block == 2:  # A10
            return (2, 0)
        elif block == 3:  # A11
            return (2, 2)
    
    # For each Strassen product q, define the A and B combinations
    # as lists of (block, sign) pairs
    
    A_combos = [
        [(0, 1), (3, 1)],     # q=0: A00 + A11
        [(2, 1), (3, 1)],     # q=1: A10 + A11
        [(0, 1)],             # q=2: A00
        [(3, 1)],             # q=3: A11
        [(0, 1), (1, 1)],     # q=4: A00 + A01
        [(2, 1), (0, -1)],    # q=5: A10 - A00
        [(1, 1), (3, -1)]     # q=6: A01 - A11
    ]
    
    B_combos = [
        [(0, 1), (3, 1)],     # q=0: B00 + B11
        [(0, 1)],             # q=1: B00
        [(2, 1), (3, -1)],    # q=2: B10 - B11
        [(1, 1), (3, -1)],    # q=3: B01 - B11
        [(3, 1)],             # q=4: B11
        [(0, 1), (1, 1)],     # q=5: B00 + B01
        [(2, 1), (3, 1)]      # q=6: B10 + B11
    ]
    
    # For each q and s, add the 2x2 Strassen factors with appropriate signs
    for q in range(7):
        for s in range(7):
            global_col = q * 7 + s
            
            # Add A factors
            for block, sign in A_combos[q]:
                ro, co = a_block_offset(block)
                for di in range(2):
                    for dj in range(2):
                        a_idx = (ro + di) * 4 + (co + dj)  # row in 4x4 A
                        u_2x2_val = U_2x2[di * 2 + dj][s]
                        U_4x4[a_idx][global_col] += sign * u_2x2_val
            
            # Add B factors
            for block, sign in B_combos[q]:
                ro, co = a_block_offset(block)
                for di in range(2):
                    for dj in range(2):
                        b_idx = (ro + di) * 4 + (co + dj)  # row in 4x4 B
                        v_2x2_val = V_2x2[di * 2 + dj][s]
                        V_4x4[b_idx][global_col] += sign * v_2x2_val
    
    # Now add W factors for the 4x4 output
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each C block (2x2), and each sub-entry (di,dj), the W factor is
    # a combination of the 7 Strassen products.
    
    C_combos = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],   # C00 = M1 + M4 - M5 + M7
        [(2, 1), (4, 1)],                     # C01 = M3 + M5
        [(1, 1), (3, 1)],                     # C10 = M2 + M4
        [(0, 1), (1, -1), (2, 1), (5, 1)]    # C11 = M1 - M2 + M3 + M6
    ]
    
    for c_block, combo in enumerate(C_combos):
        ro, co = a_block_offset(c_block)
        for di in range(2):
            for dj in range(2):
                c_idx = (ro + di) * 4 + (co + dj)
                for q, sign in combo:
                    for s in range(7):
                        global_col = q * 7 + s
                        w_2x2_val = W_2x2[di * 2 + dj][s]
                        W_4x4[c_idx][global_col] += sign * w_2x2_val
    
    # Now build the full 4x4 * 4x8 decomposition with rank 98
    # Split B into B1 (cols 0-3) and B2 (cols 4-7), each 4x4
    # Split C into C1 (cols 0-3) and C2 (cols 4-7), each 4x4
    
    U_full = [[0] * 98 for _ in range(16)]
    V_full = [[0] * 98 for _ in range(32)]
    W_full = [[0] * 98 for _ in range(32)]
    
    for r in range(49):
        # First half: A * B1 -> C1
        for i in range(16):
            U_full[i][r] = U_4x4[i][r]
            V_full[i][r] = V_4x4[i][r]  # B1 entries, rows 0-3, cols 0-3
            W_full[i][r] = W_4x4[i][r]  # C1 entries, rows 0-3, cols 0-3
        
        # Second half: A * B2 -> C2
        for i in range(16):
            U_full[i][49 + r] = U_4x4[i][r]
            V_full[i + 16][49 + r] = V_4x4[i][r]  # B2 entries, rows 0-3, cols 4-7
            W_full[i + 16][49 + r] = W_4x4[i][r]  # C2 entries, rows 0-3, cols 4-7
    
    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
