"""Rewritten program: rank-77 exact tensor decomposition for 4x4x6 matrix multiplication."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6

    # Strassen 2x2x2: 7 terms
    # Each term: (U_entries, V_entries, W_entries)
    # U entries: list of (i, j, coeff)
    # V entries: list of (j, k, coeff)
    # W entries: list of (k, i, coeff)
    strassen_2 = [
        # M1 = (A11+A22)(B11+B22)
        ([(0,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]),
        # M2 = (A21+A22)B11
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1)]),
        # M3 = A11(B12-B22)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1)]),
        # M4 = A22(B21-B11)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,1,1)]),
        # M5 = (A11+A12)B22
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(1,0,1)]),
        # M6 = (A21-A11)(B11+B12)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(0,1,1)]),
        # M7 = (A12-A22)(B21+B22)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(1,0,1)]),
    ]

    # C reconstruction for 2x2:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # W for each output position (k, i):
    # (0,0): M1(+1), M4(+1), M5(-1), M7(+1)
    # (1,0): M3(+1), M5(+1)
    # (0,1): M2(+1), M4(+1)
    # (1,1): M1(+1), M2(-1), M3(+1), M6(+1)

    def make_strassen_4x4x4():
        """Build 4x4x4 decomposition using 7 Strassen 2x2x2 instances."""
        # 4x4 matrices as 2x2 blocks of 2x2 matrices
        # Block indices: A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]]
        # Each block is 2x2.
        # We apply Strassen at the block level, where each "element" is a 2x2 matrix.
        # But since we need exact tensor decomposition (no scalar multiplication of 2x2 blocks),
        # we need to be careful: Strassen at block level would require multiplying 2x2 blocks,
        # which we can do with 8 scalar multiplications each, giving 7*8=56.
        # Alternatively, we can tile 4x4 as four 2x2x2 instances on diagonal blocks only? No, that's not correct.
        
        # Actually, for 4x4x4, the standard Strassen recursion gives rank 7*7=49.
        # This works by treating 4x4 as 2x2 blocks of 2x2 matrices, and applying Strassen
        # where each "multiplication" of 2x2 blocks is done via another Strassen (rank 7).
        
        # So we have 7 outer Strassen terms, each requiring 7 inner Strassen terms = 49 total.
        
        # For each outer term t (0..6), we have U_outer[t], V_outer[t], W_outer[t] as 2x2 blocks.
        # Each block is a 2x2 matrix, so U_outer[t] is a 2x2 matrix of 2x2 blocks? No.
        
        # Let me think again. A is 4x4, viewed as 2x2 blocks:
        # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
        # Similarly B = [[B00, B01], [B10, B11]].
        # C = AB = [[C00, C01], [C10, C11]] where each Cij is 2x2.
        
        # Strassen at block level:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each Mj is a 2x2 matrix product of 2x2 matrices.
        # To compute Mj, we use Strassen again (rank 7).
        
        # So for each outer term j, we need the U, V, W for computing (sum of some 2x2 blocks of A) * (sum of some 2x2 blocks of B).
        
        # Let me define for each outer term j:
        # U_blocks[j]: 2x2 matrix of coefficients for A's blocks (each entry is a 2x2 matrix? No, just a scalar multiplier)
        # Actually, U for the inner Strassen needs to map the 4x4 input to the 2x2 "A" input for that inner instance.
        
        # This is getting complex. Let me use a different approach:
        # Directly enumerate all 49 terms.
        
        # For outer term j (0..6), inner term k (0..6):
        # U[4x4 index, j*7+k] = U_outer[j] * U_inner[k]
        # V[4x6 index, j*7+k] = V_outer[j] * V_inner[k]  -- wait, this is 4x4x4, so V is 4x4
        # W[4x4 index, j*7+k] = W_outer[j] * W_inner[k]
        
        # But U_outer[j] is a linear combination of the 4x4 matrix A's entries.
        # Specifically, for outer term j, U_outer is a 4x4 matrix (coefficients for A).
        # U_inner[k] is a 2x2 matrix (coefficients for the 2x2 block).
        # The product U_outer[j] * U_inner[k] doesn't quite work directly.
        
        # Let me think of it differently. The 4x4 matrix A has entries A[a,b] for a,b in 0..3.
        # We partition into blocks: A00 = A[0:2,0:2], A01 = A[0:2,2:4], etc.
        
        # For outer term j, we form a 2x2 matrix S_A_j where each entry is a 2x2 block of A.
        # E.g., for j=0: S_A_0 = A00 + A11.
        # Then we need to compute S_A_0 * S_B_0 where S_B_0 = B00 + B11.
        # This is a 2x2 matrix product, which we decompose with Strassen (inner).
        
        # The inner Strassen operates on the 2x2 matrix S_A_j (entries are 2x2 blocks of A).
        # So U_inner[k] gives coefficients for the 2x2 entries of S_A_j.
        # Each entry of S_A_j is a 2x2 block of A.
        # So U_inner[k][a,b] * (block of A at position corresponding to outer term j's block (a,b))
        
        # This is very involved. Let me just hardcode or use a cleaner construction.
        
        # Alternative: use the fact that 4x4x4 can be decomposed with rank 49 by tiling.
        # I'll construct it by composing the 2x2x2 Strassen in a nested way.
        
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        # For outer term jo in 0..6, inner term ki in 0..6:
        # col = jo*7 + ki
        # U[global_A_index, col] += coeff
        # V[global_B_index, col] += coeff
        # W[global_C_index, col] += coeff
        
        # For outer term jo:
        #   U_outer[local_a, local_b] where local_a, local_b in 0..1 (block indices)
        #   These are the coefficients for the 2x2 blocks of A.
        # For inner term ki:
        #   U_inner[inner_a, inner_b] where inner_a, inner_b in 0..1 (entry indices within the 2x2 block)
        #   
        # The global A index is: block_row*2 + inner_a, block_col*2 + inner_b
        # where block_row, block_col are determined by U_outer.
        
        # U_outer[jo] is a 2x2 matrix of coefficients (for the 4 blocks of A).
        # Let me define U_outer_coeffs[jo] as a 2x2 list.
        
        U_outer_coeffs = [
            # jo=0: A00+A11
            [[1,0],[0,1]],
            # jo=1: A10+A11
            [[0,0],[1,1]],
            # jo=2: A00
            [[1,0],[0,0]],
            # jo=3: A11
            [[0,0],[0,1]],
            # jo=4: A00+A01
            [[1,1],[0,0]],
            # jo=5: A10-A00
            [[-1,0],[1,0]],
            # jo=6: A01-A11
            [[0,1],[0,-1]],
        ]
        V_outer_coeffs = [
            # jo=0: B00+B11
            [[1,0],[0,1]],
            # jo=1: B00
            [[1,0],[0,0]],
            # jo=2: B01-B11
            [[0,1],[0,-1]],
            # jo=3: B10-B00
            [[-1,0],[1,0]],
            # jo=4: B11
            [[0,0],[0,1]],
            # jo=5: B00+B01
            [[1,1],[0,0]],
            # jo=6: B10+B11
            [[0,0],[1,1]],
        ]
        
        # W for output: C = sum over jo of W_outer_coeffs[jo] * M_jo
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        # Where M_jo is the jo-th outer product result (a 2x2 matrix).
        # Each M_jo is itself a 2x2 matrix, so C has 4 blocks, each 2x2.
        
        # W_outer_coeffs[jo][block_row, block_col] = coefficient of M_jo in that block of C.
        W_outer_coeffs = [
            # jo=0: appears in C00(+1), C11(+1)
            [[1,0],[0,1]],
            # jo=1: appears in C10(+1), C11(-1)
            [[0,0],[1,-1]],
            # jo=2: appears in C01(+1), C11(+1)
            [[0,1],[0,1]],
            # jo=3: appears in C00(+1), C10(+1)
            [[1,0],[1,0]],
            # jo=4: appears in C00(-1), C01(+1)
            [[-1,1],[0,0]],
            # jo=5: appears in C11(+1)
            [[0,0],[0,1]],
            # jo=6: appears in C00(+1)
            [[1,0],[0,0]],
        ]
        
        # Now for each jo, ki:
        for jo in range(7):
            for ki in range(7):
                col = jo*7 + ki
                # U_inner[ki] is 2x2
                U_in = strassen_2[ki][0]
                V_in = strassen_2[ki][1]
                W_in = strassen_2[ki][2]
                
                # U: for each block (br, bc) in U_outer_coeffs[jo], and inner (ia, ib) in U_in:
                for br in range(2):
                    for bc in range(2):
                        coeff_outer = U_outer_coeffs[jo][br][bc]
                        if coeff_outer == 0:
                            continue
                        for ia, ib, coeff_inner in U_in:
                            global_row = br*2 + ia
                            global_col = bc*2 + ib
                            U[global_row*4 + global_col][col] += coeff_outer * coeff_inner
                
                # V: similarly for B
                for br in range(2):
                    for bc in range(2):
                        coeff_outer = V_outer_coeffs[jo][br][bc]
                        if coeff_outer == 0:
                            continue
                        for ia, ib, coeff_inner in V_in:
                            global_row = br*2 + ia
                            global_col = bc*2 + ib
                            V[global_row*4 + global_col][col] += coeff_outer * coeff_inner
                
                # W: for output C
                for br in range(2):
                    for bc in range(2):
                        coeff_outer = W_outer_coeffs[jo][br][bc]
                        if coeff_outer == 0:
                            continue
                        for ia, ib, coeff_inner in W_in:
                            global_row = br*2 + ia
                            global_col = bc*2 + ib
                            # W index: k*n+i, where k is column of C, i is row of C
                            # C's row index is global_row, column index is global_col
                            # W index = global_col * 4 + global_row
                            W[global_col*4 + global_row][col] += coeff_outer * coeff_inner
        
        return U, V, W, rank

    def make_strassen_4x4x2():
        """Build 4x4x2 decomposition using 4 Strassen 2x2x2 instances (one per 2x2x2 subproblem)."""
        # A is 4x4, B is 4x2, C is 4x2.
        # We can split into two independent 2x2x2 problems? 
        # A[0:2,0:4] * B[0:2,0:2] = C[0:2,0:2] -- no, that's not right.
        # Actually, we can split A and B by rows:
        # C[0:2, :] = A[0:2, :] * B[:, :]
        # C[2:4, :] = A[2:4, :] * B[:, :]
        # Each is a 2x4 * 4x2 = 2x2 problem? No, 2x4 * 4x2 is not 2x2x2.
        
        # Let me think: 4x4 * 4x2 = 4x2.
        # We can partition into 2x2 blocks:
        # A = [[A00, A01], [A10, A11]] (each 2x2)
        # B = [[B0], [B1]] where B0, B1 are 2x2.
        # C = [[C0], [C1]] where C0, C1 are 2x2.
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        
        # Each 2x2*2x2 = 2x2 can be done with Strassen (rank 7).
        # So we have 4 such products: A00*B0, A01*B1, A10*B0, A11*B1.
        # Total rank = 4*7 = 28.
        
        rank = 28
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(8)]
        W = [[0]*rank for _ in range(8)]
        
        # For each of the 4 sub-products, we use Strassen.
        # Sub-product 0: A00 * B0 -> contributes to C0
        # Sub-product 1: A01 * B1 -> contributes to C0
        # Sub-product 2: A10 * B0 -> contributes to C1
        # Sub-product 3: A11 * B1 -> contributes to C1
        
        for sp in range(4):
            for ki in range(7):
                col = sp*7 + ki
                U_in = strassen_2[ki][0]
                V_in = strassen_2[ki][1]
                W_in = strassen_2[ki][2]
                
                # Determine which blocks of A and B are used, and which block of C gets the result
                if sp == 0:
                    # A00 * B0 -> C0
                    a_br, a_bc = 0, 0
                    b_br = 0
                    c_br = 0
                elif sp == 1:
                    # A01 * B1 -> C0
                    a_br, a_bc = 0, 1
                    b_br = 1
                    c_br = 0
                elif sp == 2:
                    # A10 * B0 -> C1
                    a_br, a_bc = 1, 0
                    b_br = 0
                    c_br = 1
                else:
                    # A11 * B1 -> C1
                    a_br, a_bc = 1, 1
                    b_br = 1
                    c_br = 1
                
                # U: only one block of A is used
                for ia, ib, coeff in U_in:
                    global_row = a_br*2 + ia
                    global_col = a_bc*2 + ib
                    U[global_row*4 + global_col][col] += coeff
                
                # V: only one block of B is used (B is 4x2, so blocks are 2x2, indexed by br)
                for ia, ib, coeff in V_in:
                    global_row = b_br*2 + ia
                    global_col = ib
                    V[global_row*2 + global_col][col] += coeff
                
                # W: result goes to C's block c_br
                for ia, ib, coeff in W_in:
                    global_row = c_br*2 + ia
                    global_col = ib
                    # W index: k*n+i = global_col*4 + global_row
                    W[global_col*4 + global_row][col] += coeff
        
        return U, V, W, rank

    # Build 4x4x4 and 4x4x2
    U44_4, V44_4, W44_4, r1 = make_strassen_4x4x4()
    U44_2, V44_2, W44_2, r2 = make_strassen_4x4x2()
    
    total_rank = r1 + r2
    
    U_final = [[0]*total_rank for _ in range(16)]
    V_final = [[0]*total_rank for _ in range(24)]
    W_final = [[0]*total_rank for _ in range(24)]
    
    # First part: 4x4x4 (columns 0..48)
    for i in range(16):
        for c in range(r1):
            U_final[i][c] = U44_4[i][c]
    for i in range(16):
        for c in range(r1):
            V_final[i][c] = V44_4[i][c]
    for i in range(16):
        for c in range(r1):
            W_final[i][c] = W44_4[i][c]
    
    # Second part: 4x4x2 (columns 49..76)
    for i in range(16):
        for c in range(r2):
            U_final[i][r1+c] = U44_2[i][c]
    for i in range(8):
        for c in range(r2):
            V_final[i][r1+c] = V44_2[i][c]
    for i in range(8):
        for c in range(r2):
            W_final[i][r1+c] = W44_2[i][c]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
