import numpy as np
import time

DIMENSIONS = (4, 4, 7)

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28
    
    # Strategy: Use block decomposition
    # Part 1: 4x4x4 using Strassen-based rank 49
    # Part 2: 4x4x3 using optimized rank per 2x2x3 block
    # Total target: 49 + 44 = 93
    
    # Strassen 2x2x2 decomposition (7 terms)
    # Each term: (U_coeffs, V_coeffs, W_coeffs)
    # U: 2x2 matrix indices (i,j), V: 2x2 (i,j), W: 2x2 (i,j)
    strassen_2x2 = [
        # (U terms, V terms, W terms)
        # U[i,j] = sum of coeff * A[...], V[i,j] = sum of coeff * B[...], W[i,j] = sum of coeff * C[...]
        # For 2x2: A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]], C = [[c00,c01],[c10,c11]]
        # c00 = M1 - M4 + M6 - M7, c01 = M3 + M5, c10 = M2 + M4, c11 = M1 - M3 + M5 + M7
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        (
            # M1: (a00+a11)(b00+b11) -> contributes to c00(+1), c11(+1)
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ),
        (
            # M2: (a10+a11)*b00 -> contributes to c10(+1)
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        ),
        (
            # M3: a00*(b01-b11) -> contributes to c01(+1), c11(-1)
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,-1)]),
        ),
        (
            # M4: a11*(b10-b00) -> contributes to c00(-1), c10(+1)
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,-1),(1,0,1)]),
        ),
        (
            # M5: (a00+a01)*b11 -> contributes to c01(+1), c11(+1)
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(1,1,1)]),
        ),
        (
            # M6: (a10-a00)*(b00+b01) -> contributes to c00(+1)
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(0,0,1)]),
        ),
        (
            # M7: (a01-a11)*(b10+b11) -> contributes to c00(-1), c11(+1)
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,-1),(1,1,1)]),
        ),
    ]
    
    rank_444 = 49
    rank_443 = 44
    total_rank = rank_444 + rank_443
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # --- Part 1: 4x4x4 using Strassen on 2x2 blocks ---
    # 4x4 = 4 blocks of 2x2
    # A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]], C = [[C00,C01],[C10,C11]]
    # C00 = A00*B00 + A01*B10, C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10, C11 = A10*B01 + A11*B11
    # Each Ci = Aj*Bk uses 7 Strassen terms
    # Total: 4 * 7 = 28... wait, that's not right.
    # Actually for 4x4x4, we need to compute all 4 output blocks.
    # Each output block Ci is a 2x2 matrix = sum of products of 2x2 blocks.
    # C00 = A00*B00 + A01*B10: 2 Strassen decompositions = 14 terms
    # C01 = A00*B01 + A01*B11: 14 terms
    # C10 = A10*B00 + A11*B10: 14 terms
    # C11 = A10*B01 + A11*B11: 14 terms
    # Total: 56 terms? No, that's the naive way.
    # 
    # Better: Use Strassen at the 4x4 level directly.
    # 4x4 = 4 blocks of 2x2. Use 7 Strassen products of 2x2 blocks.
    # Each 2x2 product = 7 scalar products.
    # Total: 7 * 7 = 49.
    
    # For 4x4x4 Strassen:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    # C00 = S1 - S4 + S6 - S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S3 + S5 + S7
    
    # Each Si is a 2x2 product, decomposed into 7 scalar products.
    # Total: 7 * 7 = 49 scalar products.
    
    # Block positions in 4x4:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # For each Strassen term s (0-6), for each sub-term w (0-6):
    # U column: linear combination of A entries
    # V column: linear combination of B entries  
    # W column: linear combination of C entries
    
    # Strassen 4x4 terms (which blocks are used with what signs)
    strassen_4x4_terms = [
        # (A_blocks, B_blocks, C_blocks)
        # S1 = (A00+A11)(B00+B11) -> C00(+1), C11(+1)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # S2 = (A10+A11)B00 -> C10(+1)
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
        # S3 = A00(B01-B11) -> C01(+1), C11(-1)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,-1)]),
        # S4 = A11(B10-B00) -> C00(-1), C10(+1)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,-1),(1,0,1)]),
        # S5 = (A00+A01)B11 -> C01(+1), C11(+1)
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(1,1,1)]),
        # S6 = (A10-A00)(B00+B01) -> C00(+1)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(0,0,1)]),
        # S7 = (A01-A11)(B10+B11) -> C00(-1), C11(+1)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,-1),(1,1,1)]),
    ]
    
    # Block offsets: block (bi, bj) starts at row 2*bi, col 2*bj
    # A index: (2*bi + i) * m + (2*bj + j) = (2*bi + i) * 4 + (2*bj + j)
    # B index: (2*bi + i) * p + (2*bj + j) = (2*bi + i) * 7 + (2*bj + j)
    # C index: (2*bj + j) * n + (2*bi + i) = (2*bj + j) * 4 + (2*bi + i)
    
    for s_idx, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_4x4_terms):
        for w_idx, (u_def, v_def, w_def) in enumerate(strassen_2x2):
            r = s_idx * 7 + w_idx
            
            U_col = np.zeros(nm)
            V_col = np.zeros(mp)
            W_col = np.zeros(pn)
            
            # U: A entries
            for (bi, bj, ca) in a_blocks:
                for (i, j, cu) in u_def:
                    row = 2 * bi + i
                    col = 2 * bj + j
                    a_idx = row * m + col
                    U_col[a_idx] += ca * cu
            
            # V: B entries
            for (bi, bj, cb) in b_blocks:
                for (i, j, cv) in v_def:
                    row = 2 * bi + i
                    col = 2 * bj + j
                    b_idx = row * p + col
                    V_col[b_idx] += cb * cv
            
            # W: C entries
            for (bi, bj, cc) in c_blocks:
                for (i, j, cw) in w_def:
                    row = 2 * bi + i
                    col = 2 * bj + j
                    c_idx = col * n + row
                    W_col[c_idx] += cc * cw
            
            U[:, r] = U_col
            V[:, r] = V_col
            W[:, r] = W_col
    
    # --- Part 2: 4x4x3 ---
    # Compute columns 4-6 of the output (3 columns)
    # A is 4x4, B_cols is 4x3 (columns 4-6 of B)
    # Output is 4x3
    # 
    # Decompose into 2x2 blocks of A and 2x1 or 1x1 blocks of B_cols
    # A = [[A00,A01],[A10,A11]] where each is 2x2
    # B_cols = [[B0],[B1]] where B0 is 2x3, B1 is 2x3
    # 
    # C0 = A00*B0 + A01*B1 (2x3 output)
    # C1 = A10*B0 + A11*B1 (2x3 output)
    #
    # For 2x2 * 2x3: 
    # A = [[a0,a1],[a2,a3]], B = [[b0,b1,b2],[c0,c1,c2]]
    # C = [[a0*b0+a1*c0, a0*b1+a1*c1, a0*b2+a1*c2],
    #      [a2*b0+a3*c0, a2*b1+a3*c1, a2*b2+a3*c2]]
    #
    # Rank of 2x2x3 tensor: minimum is 5? Let me think...
    # Actually, the border rank of matrix multiplication [2,2,3] is...
    # For 2x2x3, the tensor rank is at most 2*3=6 (schoolbook).
    # Can we do better? 
    # 
    # Using Strassen-like: 
    # The 2x2x3 problem: A(2x2) * B(2x3) -> C(2x3)
    # This is equivalent to multiplying a 2x2 matrix by a 2x3 matrix.
    # 
    # One approach: Use 5 or 6 terms.
    # Let's try rank 5:
    # Actually, I recall that R([2,2,3]) = 5 or 6.
    # Let me use a known decomposition.
    
    # For 2x2x3, let's use this decomposition (rank 6, schoolbook-like but optimized):
    # Actually, let me just use the straightforward approach with some sharing.
    
    # 2x2x3 decomposition (rank 6):
    # Each element of C is a bilinear form.
    # C[0,0] = a0*b0 + a1*c0
    # C[0,1] = a0*b1 + a1*c1
    # C[0,2] = a0*b2 + a1*c2
    # C[1,0] = a2*b0 + a3*c0
    # C[1,1] = a2*b1 + a3*c1
    # C[1,2] = a2*b2 + a3*c2
    
    # We can factor:
    # M1 = a0*b0, M2 = a1*c0 -> C[0,0] = M1+M2
    # M3 = a0*b1, M4 = a1*c1 -> C[0,1] = M3+M4
    # M5 = a0*b2, M6 = a1*c2 -> C[0,2] = M5+M6
    # M7 = a2*b0, M8 = a3*c0 -> C[1,0] = M7+M8
    # M9 = a2*b1, M10 = a3*c1 -> C[1,1] = M9+M10
    # M11 = a2*b2, M12 = a3*c2 -> C[1,2] = M11+M12
    # That's rank 12, too high.
    
    # Better: share the a0 and a1 factors:
    # M1 = a0*(b0+b1+b2), M2 = a1*(c0+c1+c2)
    # C[0,0] = a0*b0 + a1*c0 (need separate terms)
    
    # Let me use a different approach. For 2x2x3:
    # Think of it as: for each column of B (3 columns), do a 2x2x1 multiplication.
    # 2x2x1: A(2x2)*b(2x1) -> c(2x1)
    # c0 = a0*b0 + a1*b1
    # c1 = a2*b0 + a3*b1
    # Rank of 2x2x1 = 2 (schoolbook is optimal for this)
    # So 3 columns * 2 = 6 terms.
    
    # But we can do better by sharing across columns!
    # M1 = a0*(b0[0]+b1[0]+b2[0]), etc.
    
    # Actually, let me use the known result: R([2,2,3]) = 5
    # Here's a rank-5 decomposition:
    # 
    # After research, the border rank is 5 but tensor rank might be 6.
    # Let me use rank 6 to be safe, or try rank 5.
    
    # For simplicity and correctness, let me use rank 6 per 2x2x3 block.
    # 4 blocks (2x2 in A) * 6 = 24... no wait.
    # 
    # A is 4x4 = 4 blocks of 2x2. B_cols is 4x3 = 2 blocks of 2x3.
    # C is 4x3 = 2 blocks of 2x3.
    # C_top = A00*B0 + A01*B1 (2x3)
    # C_bot = A10*B0 + A11*B1 (2x3)
    # 
    # Each Ci = Aj*Bk + Al*Bm involves two 2x2x3 multiplications.
    # If each 2x2x3 has rank r, then each Ci has rank 2r.
    # Total: 2 * 2r = 4r.
    # With r=6: total = 24. But we need 44... 
    # 
    # Wait, I think I'm confusing myself. Let me recount.
    # 
    # The current program uses 11 per 2x2x3 block, with 4 blocks = 44.
    # Let me verify: 4 subproblems * 11 = 44. Yes.
    # 
    # Can I do better than 11 per 2x2x3 block?
    # A 2x2x3 block: A(2x2) * B(2x3) -> C(2x3)
    # Schoolbook: 2*2*3 = 12 terms.
    # The current uses 11, which is already an improvement.
    # 
    # Can we do rank 10 or less?
    # 
    # Let me try a different decomposition for 2x2x3.
    # 
    # Actually, I realize the "11 per 2x2x3 block" in the current code might be 
    # using Strassen(7) + schoolbook(4) = 11. Let me verify this makes sense.
    # 
    # For 2x2x3: 
    # Use Strassen on the first 2 columns of B (treating them as 2x2):
    # That gives 7 terms for the 2x2x2 part.
    # Then 4 terms for the remaining column (2x2x1, but 2x2x1 has rank 2, so 2 terms).
    # Total: 7 + 2 = 9? Or 7 + 4 = 11?
    # 
    # 2x2x1: A(2x2)*b(2x1) -> c(2x1)
    # c0 = a00*b0 + a01*b1
    # c1 = a10*b0 + a11*b1
    # This needs 2 terms: M1=a00*b0, M2=a01*b1 for c0; M3=a10*b0, M4=a11*b1 for c1.
    # But we can share: M1=a00*b0, M2=a01*b1, M3=a10*b0, M4=a11*b1 = 4 terms.
    # Or: M1=a00*b0, M2=a01*b1 -> c0=M1+M2; M3=a10*b0, M4=a11*b1 -> c1=M3+M4. Still 4.
    # Actually 2x2x1 has rank 2: 
    # M1 = a00*b0, M2 = (a01+a10+a11)*b1? No, that doesn't work.
    # 
    # 2x2x1 tensor rank: The tensor is c0 = a00*b0 + a01*b1, c1 = a10*b0 + a11*b1.
    # This is rank 2 (you need at least 2 terms since it's not a single product).
    # M1 = a00*b0 -> contributes to c0
    # M2 = a01*b1 -> contributes to c0
    # M3 = a10*b0 -> contributes to c1  
    # M4 = a11*b1 -> contributes to c1
    # That's 4 terms. Can we do 2?
    # M1 = (a00+a10)*b0 -> contributes a00*b0 to c0 and a10*b0 to c1
    # M2 = (a01+a11)*b1 -> contributes a01*b1 to c0 and a11*b1 to c1
    # Yes! Rank 2.
    # 
    # So 2x2x3: Strassen(7) for 2x2x2 + rank-2 for 2x2x1 = 9 terms.
    # 4 blocks * 9 = 36. Total: 49 + 36 = 85.
    
    # Let me implement this!
    
    # Reset and use rank 85
    rank_443_new = 36
    total_rank_new = rank_444 + rank_443_new  # 49 + 36 = 85
    
    U = np.zeros((nm, total_rank_new), dtype=float)
    V = np.zeros((mp, total_rank_new), dtype=float)
    W = np.zeros((pn, total_rank_new), dtype=float)
    
    # Re-do Part 1 (same as before)
    for s_idx, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_4x4_terms):
        for w_idx, (u_def, v_def, w_def) in enumerate(strassen_2x2):
            r = s_idx * 7 + w_idx
            
            U_col = np.zeros(nm)
            V_col = np.zeros(mp)
            W_col = np.zeros(pn)
            
            for (bi, bj, ca) in a_blocks:
                for (i, j, cu) in u_def:
                    row = 2 * bi + i
                    col = 2 * bj + j
                    a_idx = row * m + col
                    U_col[a_idx] += ca * cu
            
            for (bi, bj, cb) in b_blocks:
                for (i, j, cv) in v_def:
                    row = 2 * bi + i
                    col = 2 * bj + j
                    b_idx = row * p + col
                    V_col[b_idx] += cb * cv
            
            for (bi, bj, cc) in c_blocks:
                for (i, j, cw) in w_def:
                    row = 2 * bi + i
                    col = 2 * bj + j
                    c_idx = col * n + row
                    W_col[c_idx] += cc * cw
            
            U[:, r] = U_col
            V[:, r] = V_col
            W[:, r] = W_col
    
    # Part 2: 4x4x3 with rank 36
    # 4 subproblems: each is (2x2 A-block) * (2x3 B-block) -> (2x3 C-block)
    # Subproblem (ar, ac): A rows ar:ar+2, cols ac:ac+2; B rows ar:ar+2, cols 4:7; C rows ar:ar+2, cols 4:7
    
    offset = rank_444
    col = 0
    
    subproblems = [
        (0, 0),  # A[0:2,0:2] * B[0:2,4:7] -> C[0:2,4:7]
        (0, 2),  # A[0:2,2:4] * B[0:2,4:7] -> C[0:2,4:7]
        (2, 0),  # A[2:4,0:2] * B[2:4,4:7] -> C[2:4,4:7]
        (2, 2),  # A[2:4,2:4] * B[2:4,4:7] -> C[2:4,4:7]
    ]
    
    # For each subproblem, decompose 2x2x3 as:
    # Strassen(7) for columns 4,5 of B (2x2x2 part)
    # Rank-2 for column 6 of B (2x2x1 part)
    # Total: 9 terms per subproblem
    
    for (ar, ac) in subproblems:
        # Strassen part: A[ar:ar+2, ac:ac+2] * B[ar:ar+2, 4:6] -> C[ar:ar+2, 4:6]
        for s_idx, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_2x2):
            for w_idx, (u_def, v_def, w_def) in enumerate(strassen_2x2):
                U_col = np.zeros(nm)
                V_col = np.zeros(mp)
                W_col = np.zeros(pn)
                
                # U: A entries
                for (bi, bj, ca) in a_blocks:
                    for (i, j, cu) in u_def:
                        row = ar + bi
                        col_a = ac + bj
                        a_idx = row * m + col_a
                        U_col[a_idx] += ca * cu
                
                # V: B entries (columns 4,5)
                for (bi, bj, cb) in b_blocks:
                    for (i, j, cv) in v_def:
                        row = ar + bi
                        col_b = 4 + bj  # columns 4,5
                        b_idx = row * p + col_b
                        V_col[b_idx] += cb * cv
                
                # W: C entries (columns 4,5)
                for (bi, bj, cc) in c_blocks:
                    for (i, j, cw) in w_def:
                        row = ar + bi
                        col_c = 4 + bj  # columns 4,5
                        c_idx = col_c * n + row
                        W_col[c_idx] += cc * cw
                
                U[:, offset + col] = U_col
                V[:, offset + col] = V_col
                W[:, offset + col] = W_col
                col += 1
        
        # Rank-2 part: A[ar:ar+2, ac:ac+2] * B[ar:ar+2, 6] -> C[ar:ar+2, 6]
        # M1 = (A[ar,ac]+A[ar+1,ac]) * B[ar,6]
        # M2 = (A[ar,ac+1]+A[ar+1,ac+1]) * B[ar+1,6]
        # C[ar,6] = M1 + M2... wait, let me be more careful.
        # 
        # A = [[a00,a01],[a10,a11]], b = [b0,b1]^T
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        # 
        # M1 = (a00+a10)*b0 -> contributes a00*b0 to c0, a10*b0 to c1
        # M2 = (a01+a11)*b1 -> contributes a01*b1 to c0, a11*b1 to c1
        # c0 = M1 + M2, c1 = M1 + M2? No!
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        # M1 = (a00+a10)*b0 = a00*b0 + a10*b0
        # M2 = (a01+a11)*b1 = a01*b1 + a11*b1
        # c0 = a00*b0 + a01*b1 (not M1+M2)
        # 
        # I need: c0 = a00*b0 + a01*b1, c1 = a10*b0 + a11*b1
        # M1 = a00*b0, M2 = a01*b1 -> c0 = M1+M2
        # M3 = a10*b0, M4 = a11*b1 -> c1 = M3+M4
        # That's 4 terms, not 2.
        # 
        # Can I share? 
        # M1 = a00*b0, M2 = a10*b0 -> both use b0
        # M3 = a01*b1, M4 = a11*b1 -> both use b1
        # c0 = M1 + M3, c1 = M2 + M4
        # Still 4 terms.
        # 
        # Actually, the rank of 2x2x1 is 2, not 4. Let me reconsider.
        # The tensor T = c0*(a00*b0 + a01*b1) + c1*(a10*b0 + a11*b1)
        # = a00*b0*c0 + a01*b1*c0 + a10*b0*c1 + a11*b1*c1
        # 
        # Can I write this as 2 terms?
        # M1 = (a00*c0 + a10*c1)*b0
        # M2 = (a01*c0 + a11*c1)*b1
        # Yes! Rank 2.
        
        # Term 1: U = (a00*c0_coeff + a10*c1_coeff), V = b0, W = 1
        # For our case: 
        # A[ar,ac] corresponds to a00, A[ar+1,ac] to a10
        # A[ar,ac+1] to a01, A[ar+1,ac+1] to a11
        # B[ar,6] to b0, B[ar+1,6] to b1
        # C[ar,6] to c0, C[ar+1,6] to c1
        
        # Term 1: M1 = (A[ar,ac]*1 + A[ar+1,ac]*1) * B[ar,6]
        # U: A[ar,ac]=1, A[ar+1,ac]=1
        # V: B[ar,6]=1
        # W: C[ar,6]=1, C[ar+1,6]=1
        
        U_col = np.zeros(nm)
        V_col = np.zeros(mp)
        W_col = np.zeros(pn)
        
        U_col[(ar) * m + ac] = 1.0
        U_col[(ar+1) * m + ac] = 1.0
        V_col[ar * p + 6] = 1.0
        W_col[6 * n + ar] = 1.0
        W_col[6 * n + ar+1] = 1.0
        
        U[:, offset + col] = U_col
        V[:, offset + col] = V_col
        W[:, offset + col] = W_col
        col += 1
        
        # Term 2: M2 = (A[ar,ac+1]*1 + A[ar+1,ac+1]*1) * B[ar+1,6]
        U_col = np.zeros(nm)
        V_col = np.zeros(mp)
        W_col = np.zeros(pn)
        
        U_col[(ar) * m + (ac+1)] = 1.0
        U_col[(ar+1) * m + (ac+1)] = 1.0
        V_col[(ar+1) * p + 6] = 1.0
        W_col[6 * n + ar] = 1.0
        W_col[6 * n + ar+1] = 1.0
        
        U[:, offset + col] = U_col
        V[:, offset + col] = V_col
        W[:, offset + col] = W_col
        col += 1
    
    # Verify col = total_rank_new
    assert col == total_rank_new, f"col={col}, total_rank_new={total_rank_new}"
    
    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
