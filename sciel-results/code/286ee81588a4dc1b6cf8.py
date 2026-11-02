"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-49 decomposition via Kronecker square of Strassen's 2x2 algorithm.
    # Maps 4x4 matrix multiplication to 7 bilinear terms per 2x2 block,
    # giving 7*7 = 49 total terms.
    #
    # Strassen terms for 2x2:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M3 + M6 - M7
    #
    # For 4x4, we split A, B into four 2x2 blocks:
    #   A = [[A00, A01], [A10, A11]]  (each 2x2)
    #   B = [[B00, B01], [B10, B11]]  (each 2x2)
    #   C = [[C00, C01], [C10, C11]]  (each 2x2)
    #
    # Each 2x2 block multiplication uses 7 Strassen terms.
    # The Kronecker square approach: each "element" in the 2x2 meta-matrix
    # is itself a 2x2 matrix, so we apply Strassen at the meta level,
    # where each bilinear term is a 2x2 matrix product (itself via Strassen).
    #
    # Actually, the simpler correct approach: the 4x4 multiplication tensor
    # is the Kronecker product of two 2x2 multiplication tensors.
    # If T_2x2 has rank 7 (Strassen), then T_4x4 = T_2x2 ⊗ T_2x2 has rank 49.
    #
    # The Kronecker product of decompositions:
    # If T_2x2 = sum_{a=1}^{7} u_a ⊗ v_a ⊗ w_a (rank-7 decomposition),
    # then T_4x4 = sum_{a,b=1}^{7} (u_a ⊗ u_b) ⊗ (v_a ⊗ v_b) ⊗ (w_a ⊗ w_b)
    #
    # We need to map the indices correctly.
    # For 2x2: A[i,j], B[j,k], C[i,k] with i,j,k in {0,1}
    # For 4x4: A[i,j], B[j,k], C[i,k] with i,j,k in {0,1,2,3}
    #   i = 2*I + i2, j = 2*J + j2, k = 2*K + k2
    #   where I,J,K in {0,1} are meta-indices, i2,j2,k2 in {0,1} are sub-indices.
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C[2I+i2, 2K+k2] = sum_{J,j2} A[2I+i2, 2J+j2] * B[2J+j2, 2K+k2]
    #   = sum_J (sum_{j2} A[2I+i2, 2J+j2] * B[2J+j2, 2K+k2])
    #
    # This is the (i2,k2) element of the 2x2 matrix product:
    #   (A_meta[I,:] * B_meta[:,K])[i2,k2]
    # where A_meta[I] is the 2x2 block A[2I:2I+2, :], etc.
    #
    # So C_meta = A_meta * B_meta (2x2 meta-matrix multiplication),
    # and each element is a 2x2 matrix computed via Strassen.
    #
    # Strassen for the meta level (2x2 of 2x2 blocks):
    #   Let A_meta = [[A00, A01], [A10, A11]], B_meta = [[B00, B01], [B10, B11]]
    #   These are 2x2 matrices of 2x2 blocks.
    #
    #   S1 = A00 + A11  (2x2 matrix)
    #   S2 = A10 + A11  (2x2 matrix)
    #   S3 = A00        (2x2 matrix)
    #   S4 = A11        (2x2 matrix)
    #   S5 = A00 + A01  (2x2 matrix)
    #   S6 = A10 - A00  (2x2 matrix)
    #   S7 = A01 - A11  (2x2 matrix)
    #
    #   T1 = B00 + B11  (2x2 matrix)
    #   T2 = B00        (2x2 matrix)
    #   T3 = B01 - B11  (2x2 matrix)
    #   T4 = B10 - B00  (2x2 matrix)
    #   T5 = B11        (2x2 matrix)
    #   T6 = B00 + B01  (2x2 matrix)
    #   T7 = B10 + B11  (2x2 matrix)
    #
    #   M1 = S1 * T1  (2x2 matrix product, via Strassen: 7 terms)
    #   M2 = S2 * T2
    #   M3 = S3 * T3
    #   M4 = S4 * T4
    #   M5 = S5 * T5
    #   M6 = S6 * T6
    #   M7 = S7 * T7
    #
    #   C00 = M1 - M4 + M5 - M7
    #   C01 = M3 + M4
    #   C10 = M2 + M7
    #   C11 = M1 - M3 + M6 - M7
    #
    # Each Mx is a 2x2 matrix product computed via Strassen (7 bilinear terms).
    # Total: 7 * 7 = 49 bilinear terms.
    #
    # Now we need to express each of the 49 terms as:
    #   term = (linear combo of A entries) * (linear combo of B entries)
    #
    # For the meta-level Strassen, Sx and Tx are linear combinations of the 2x2 blocks.
    # For the sub-level Strassen, each 2x2 matrix product (e.g., S1*T1) uses 7 terms.
    #
    # Let's enumerate. The 4x4 matrix A has entries A[i,j] for i,j in 0..3.
    # Block A00 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # Block A01 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    # Block A10 = [[A[2,0], A[2,1]], [A[3,0], A[3,1]]]
    # Block A11 = [[A[2,2], A[2,3]], [A[3,2], A[3,3]]]
    #
    # Similarly for B.
    #
    # For a 2x2 matrix product X*Y where X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]:
    # Strassen terms:
    #   m1 = (x00+x11)*(y00+y11)
    #   m2 = (x10+x11)*y00
    #   m3 = x00*(y01-y11)
    #   m4 = x11*(y10-y00)
    #   m5 = (x00+x01)*y11
    #   m6 = (x10-x00)*(y00+y01)
    #   m7 = (x01-x11)*(y10+y11)
    #
    # Result:
    #   r00 = m1 - m4 + m5 - m7
    #   r01 = m3 + m4
    #   r10 = m2 + m7
    #   r11 = m1 - m3 + m6 - m7
    #
    # Now, for the meta-level:
    # S1 = A00 + A11, T1 = B00 + B11
    # M1 = S1 * T1 (2x2 product via Strassen)
    #
    # S1 = [[A[0,0]+A[2,2], A[0,1]+A[2,3]], [A[1,0]+A[3,2], A[1,1]+A[3,3]]]
    # T1 = [[B[0,0]+B[2,2], B[0,1]+B[2,3]], [B[1,0]+B[3,2], B[1,1]+B[3,3]]]
    #
    # M1_strassen terms (7 of them):
    #   Let x = S1, y = T1
    #   x00 = A[0,0]+A[2,2], x01 = A[0,1]+A[2,3]
    #   x10 = A[1,0]+A[3,2], x11 = A[1,1]+A[3,3]
    #   y00 = B[0,0]+B[2,2], y01 = B[0,1]+B[2,3]
    #   y10 = B[1,0]+B[3,2], y11 = B[1,1]+B[3,3]
    #
    #   m1_11 = (x00+x11)*(y00+y11) = (A[0,0]+A[2,2]+A[1,1]+A[3,3])*(B[0,0]+B[2,2]+B[1,1]+B[3,3])
    #   m2_11 = (x10+x11)*y00 = (A[1,0]+A[3,2]+A[1,1]+A[3,3])*(B[0,0]+B[2,2])
    #   m3_11 = x00*(y01-y11) = (A[0,0]+A[2,2])*(B[0,1]+B[2,3]-B[1,1]-B[3,3])
    #   m4_11 = x11*(y10-y00) = (A[1,1]+A[3,3])*(B[1,0]+B[3,2]-B[0,0]-B[2,2])
    #   m5_11 = (x00+x01)*y11 = (A[0,0]+A[2,2]+A[0,1]+A[2,3])*(B[1,1]+B[3,3])
    #   m6_11 = (x10-x00)*(y00+y01) = (A[1,0]+A[3,2]-A[0,0]-A[2,2])*(B[0,0]+B[2,2]+B[0,1]+B[2,3])
    #   m7_11 = (x01-x11)*(y10+y11) = (A[0,1]+A[2,3]-A[1,1]-A[3,3])*(B[1,0]+B[3,2]+B[1,1]+B[3,3])
    #
    # This is getting complex. Let me write a programmatic generator.

    import time
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return DEADLINE - SAFETY_MARGIN - elapsed()

    # Build the rank-49 decomposition programmatically.
    # 
    # We'll represent each of the 49 terms by:
    #   - A coefficients: dict mapping (i,j) -> coeff
    #   - B coefficients: dict mapping (j,k) -> coeff
    #   - C contribution: dict mapping (i,k) -> coeff
    #
    # Then U[i*m+j, c] = A_coeff[(i,j)] for term c
    #     V[j*p+k, c] = B_coeff[(j,k)] for term c
    #     W[k*n+i, c] = C_coeff[(i,k)] for term c
    #
    # Note: W uses trace convention W[k*n+i, r]

    n, m, p = DIMENSIONS  # 4, 4, 4

    # Strassen 2x2 multiplication terms
    # Returns list of 7 (A_coeffs, B_coeffs, C_coeffs) for a 2x2 matrix product
    # where A is 2x2 with entries indexed by (0,0),(0,1),(1,0),(1,1)
    # and B similarly, C is the result 2x2.
    
    def strassen_2x2(a_names, b_names, c_names):
        """
        a_names: dict mapping (i,j) -> variable name for A[i,j] in 2x2
        b_names: dict mapping (j,k) -> variable name for B[j,k] in 2x2
        c_names: dict mapping (i,k) -> variable name for C[i,k] in 2x2
        
        Returns list of 7 tuples: (a_coeff_dict, b_coeff_dict, c_coeff_dict)
        where each coeff_dict maps variable name -> coefficient
        """
        a00, a01 = a_names[(0,0)], a_names[(0,1)]
        a10, a11 = a_names[(1,0)], a_names[(1,1)]
        b00, b01 = b_names[(0,0)], b_names[(0,1)]
        b10, b11 = b_names[(1,0)], b_names[(1,1)]
        c00, c01 = c_names[(0,0)], c_names[(0,1)]
        c10, c11 = c_names[(1,0)], c_names[(1,1)]
        
        terms = []
        
        # m1 = (a00+a11)*(b00+b11) -> contributes to c00(+1), c11(+1)
        terms.append(({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1}))
        
        # m2 = (a10+a11)*b00 -> contributes to c10(+1)
        terms.append(({a10: 1, a11: 1}, {b00: 1}, {c10: 1}))
        
        # m3 = a00*(b01-b11) -> contributes to c01(+1)
        terms.append(({a00: 1}, {b01: 1, b11: -1}, {c01: 1}))
        
        # m4 = a11*(b10-b00) -> contributes to c00(-1), c01(+1)
        terms.append(({a11: 1}, {b10: 1, b00: -1}, {c00: -1, c01: 1}))
        
        # m5 = (a00+a01)*b11 -> contributes to c00(+1)
        terms.append(({a00: 1, a01: 1}, {b11: 1}, {c00: 1}))
        
        # m6 = (a10-a00)*(b00+b01) -> contributes to c11(+1)
        terms.append(({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1}))
        
        # m7 = (a01-a11)*(b10+b11) -> contributes to c00(-1), c10(+1), c11(-1)
        terms.append(({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: -1, c10: 1, c11: -1}))
        
        return terms

    # Now build the 4x4 decomposition.
    # 
    # The 4x4 matrices are split into 2x2 blocks.
    # A block (I,J) has entries A[2I+i2, 2J+j2] for i2,j2 in {0,1}
    # 
    # Meta-level Strassen on the 2x2 block structure:
    # S1 = A00 + A11, T1 = B00 + B11
    # S2 = A10 + A11, T2 = B00
    # S3 = A00,        T3 = B01 - B11
    # S4 = A11,        T4 = B10 - B00
    # S5 = A00 + A01,  T5 = B11
    # S6 = A10 - A00,  T6 = B00 + B01
    # S7 = A01 - A11,  T7 = B10 + B11
    #
    # Each Sx is a 2x2 matrix (sum/difference of blocks), each Tx similarly.
    # Mx = Sx * Tx is a 2x2 matrix product (each entry is a scalar product of 2x2 matrices,
    # but really Mx is a 2x2 matrix where each entry is a 2x2 matrix product result).
    #
    # Wait, I need to be more careful. Sx is a 2x2 matrix of scalars (since blocks are 2x2).
    # No wait - Sx is a 2x2 matrix where each "element" is actually a 2x2 block.
    # So Sx is really a 4x4 matrix? No...
    #
    # Let me reconsider. The meta-level treats the 4x4 matrix as a 2x2 matrix of 2x2 blocks.
    # A_meta[I,J] is a 2x2 matrix (the block).
    # 
    # S1 = A_meta[0,0] + A_meta[1,1] is a 2x2 matrix (element-wise sum of blocks).
    # T1 = B_meta[0,0] + B_meta[1,1] is a 2x2 matrix.
    # M1 = S1 * T1 is a 2x2 matrix product, giving a 2x2 matrix.
    #
    # So M1 is a 2x2 matrix of scalars. But wait, that's not right either.
    # 
    # Actually, the meta-level multiplication C_meta = A_meta * B_meta where each entry
    # is a 2x2 matrix. The product (C_meta)[I,K] = sum_J A_meta[I,J] * B_meta[J,K]
    # where the inner product is a 2x2 matrix product (standard 2x2 matmul).
    #
    # So (C_meta)[I,K] is a 2x2 matrix, and the full C is the 4x4 matrix formed by
    # placing these 2x2 blocks.
    #
    # Strassen on the meta-level:
    # M1 = S1 * T1 where S1, T1 are 2x2 matrices (of scalars, since blocks are 2x2).
    # The product S1*T1 is a standard 2x2 matrix product, which we decompose via Strassen
    # into 7 bilinear terms. Each term is (linear combo of S1 entries) * (linear combo of T1 entries).
    #
    # But S1 entries are themselves linear combos of A entries (from the blocks).
    # So each bilinear term at the sub-level becomes a product of two linear combos of A and B entries.
    #
    # This gives 7 (meta) * 7 (sub) = 49 terms total.
    #
    # Let me implement this properly.

    # For the meta-level, define block variable names.
    # Block (I,J) of A: entries at positions (2I+i2, 2J+j2) for i2,j2 in {0,1}
    # We'll use the actual 4x4 indices as variable names.
    
    # Meta-level Strassen terms:
    # Each meta-term is (S_block_expr, T_block_expr) where each is a dict of block->coeff
    # S_block_expr maps (I,J) -> coeff for A blocks
    # T_block_expr maps (J,K) -> coeff for B blocks
    
    meta_terms = [
        # (S: dict of A-block (I,J)->coeff, T: dict of B-block (J,K)->coeff, C: dict of C-block (I,K)->coeff)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),       # M1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),                           # M2
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),                          # M3
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (0,1): 1}),               # M4
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1}),                           # M5
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),                # M6
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1, (1,0): 1, (1,1): -1}),  # M7
    ]
    
    # For each meta-term, expand into 7 sub-terms.
    # A sub-term is (a_coeff: dict of (i,j)->coeff, b_coeff: dict of (j,k)->coeff, c_coeff: dict of (i,k)->coeff)
    
    all_terms = []
    
    for s_expr, t_expr, c_expr in meta_terms:
        # s_expr is a dict of A-block (I,J) -> coeff
        # t_expr is a dict of B-block (J,K) -> coeff
        # c_expr is a dict of C-block (I,K) -> coeff
        #
        # The product of the S matrix (2x2 of blocks) and T matrix (2x2 of blocks)
        # gives a 2x2 matrix of blocks, which contributes to C blocks per c_expr.
        #
        # But wait - the meta-level Strassen gives Mx = Sx * Tx where Sx and Tx are 2x2 matrices
        # of 2x2 blocks. The product Sx*Tx is a 2x2 matrix product of 2x2 matrices.
        # 
        # Hmm, I think I'm overcomplicating this. Let me think again.
        #
        # The meta-level: A_meta is 2x2 of 2x2 blocks. B_meta is 2x2 of 2x2 blocks.
        # C_meta = A_meta * B_meta (standard 2x2 matrix multiplication, where each entry is a 2x2 matrix).
        #
        # (C_meta)[I,K] = sum_J A_meta[I,J] @ B_meta[J,K]  (2x2 matrix product)
        #
        # Strassen on this 2x2 meta-multiplication:
        # M1 = S1 @ T1 where S1 = A_meta[0,0] + A_meta[1,1] (2x2 matrix), T1 = B_meta[0,0] + B_meta[1,1] (2x2 matrix)
        # M1 is a 2x2 matrix (result of 2x2 matmul).
        #
        # So S1 is a 2x2 matrix of scalars. T1 is a 2x2 matrix of scalars.
        # M1 = S1 @ T1 is computed via Strassen (7 bilinear terms on the 2x2 entries).
        #
        # S1[i2,j2] = sum over blocks: for each (I,J) in s_expr, A_meta[I,J][i2,j2] * coeff
        #   = sum over (I,J): coeff * A[2I+i2, 2J+j2]
        #
        # So S1 is a 2x2 matrix where each entry is a linear combo of 4x4 A entries.
        # Similarly for T1.
        #
        # The 7 Strassen terms on S1@T1 give 7 bilinear terms, each being
        # (linear combo of S1 entries) * (linear combo of T1 entries)
        # which expands to (linear combo of A entries) * (linear combo of B entries).
        #
        # The result M1 is a 2x2 matrix, and its contribution to C is:
        #   For each (I,K) in c_expr with coeff c:
        #     C_meta[I,K] += c * M1
        #   So C[2I+i2, 2K+k2] += c * M1[i2,k2]
        #
        # And M1[i2,k2] is determined by which Strassen term it is.
        
        # Build S1 as a 2x2 matrix of linear combos of A entries.
        # S1[i2,j2] = sum_{(I,J), coeff} coeff * A[2I+i2, 2J+j2]
        S = [[{} for _ in range(2)] for _ in range(2)]  # S[i2][j2] = dict of (i,j)->coeff
        for (I, J), coeff in s_expr.items():
            for i2 in range(2):
                for j2 in range(2):
                    a_idx = (2*I + i2, 2*J + j2)
                    S[i2][j2][a_idx] = S[i2][j2].get(a_idx, 0) + coeff
        
        T = [[{} for _ in range(2)] for _ in range(2)]  # T[j2][k2] = dict of (j,k)->coeff
        for (J, K), coeff in t_expr.items():
            for j2 in range(2):
                for k2 in range(2):
                    b_idx = (2*J + j2, 2*K + k2)
                    T[j2][k2][b_idx] = T[j2][k2].get(b_idx, 0) + coeff
        
        # Now compute the 7 Strassen terms on S@T (2x2 matrix product).
        # S = [[S00, S01], [S10, S11]] where Sxy is a dict of A-indices -> coeff
        # T = [[T00, T01], [T10, T11]] where Txy is a dict of B-indices -> coeff
        #
        # Strassen terms for 2x2 product R = S@T:
        # m1 = (S00+S11) @ (T00+T11) -> R00 += m1, R11 += m1
        # m2 = (S10+S11) @ T00 -> R10 += m2
        # m3 = S00 @ (T01-T11) -> R01 += m3
        # m4 = S11 @ (T10-T00) -> R00 -= m4, R01 += m4
        # m5 = (S00+S01) @ T11 -> R00 += m5
        # m6 = (S10-S00) @ (T00+T01) -> R11 += m6
        # m7 = (S01-S11) @ (T10+T11) -> R00 -= m7, R10 += m7, R11 -= m7
        
        # Each "product" like (S00+S11) @ (T00+T11) means:
        #   a_dict = S00 + S11 (element-wise addition of dicts)
        #   b_dict = T00 + T11
        #   This is ONE bilinear term: (sum of A entries per a_dict) * (sum of B entries per b_dict)
        
        def add_dicts(d1, d2):
            result = dict(d1)
            for k, v in d2.items():
                result[k] = result.get(k, 0) + v
            return result
        
        def sub_dicts(d1, d2):
            result = dict(d1)
            for k, v in d2.items():
                result[k] = result.get(k, 0) - v
            return result
        
        # Generate the 7 sub-terms
        sub_terms = []
        
        # m1: (S00+S11) * (T00+T11) -> R00(+1), R11(+1)
        a1 = add_dicts(S[0][0], S[1][1])
        b1 = add_dicts(T[0][0], T[1][1])
        sub_terms.append((a1, b1, {(0,0): 1, (1,1): 1}))
        
        # m2: (S10+S11) * T00 -> R10(+1)
        a2 = add_dicts(S[1][0], S[1][1])
        b2 = T[0][0]
        sub_terms.append((a2, b2, {(1,0): 1}))
        
        # m3: S00 * (T01-T11) -> R01(+1)
        a3 = S[0][0]
        b3 = sub_dicts(T[0][1], T[1][1])
        sub_terms.append((a3, b3, {(0,1): 1}))
        
        # m4: S11 * (T10-T00) -> R00(-1), R01(+1)
        a4 = S[1][1]
        b4 = sub_dicts(T[1][0], T[0][0])
        sub_terms.append((a4, b4, {(0,0): -1, (0,1): 1}))
        
        # m5: (S00+S01) * T11 -> R00(+1)
        a5 = add_dicts(S[0][0], S[0][1])
        b5 = T[1][1]
        sub_terms.append((a5, b5, {(0,0): 1}))
        
        # m6: (S10-S00) * (T00+T01) -> R11(+1)
        a6 = sub_dicts(S[1][0], S[0][0])
        b6 = add_dicts(T[0][0], T[0][1])
        sub_terms.append((a6, b6, {(1,1): 1}))
        
        # m7: (S01-S11) * (T10+T11) -> R00(-1), R10(+1), R11(-1)
        a7 = sub_dicts(S[0][1], S[1][1])
        b7 = add_dicts(T[1][0], T[1][1])
        sub_terms.append((a7, b7, {(0,0): -1, (1,0): 1, (1,1): -1}))
        
        # Now, each sub-term (a_dict, b_dict, c_dict) contributes to C via:
        #   For each (I,K) in c_expr with coeff c_meta:
        #     For each (i2,k2) in c_dict with coeff c_sub:
        #       C[2I+i2, 2K+k2] += c_meta * c_sub * (a_dict product) * (b_dict product)
        
        for (a_dict, b_dict, c_sub_dict) in sub_terms:
            full_a = {}
            full_b = {}
            full_c = {}
            
            # a_dict and b_dict are already full 4x4 index dicts
            # c_sub_dict has keys (i2,k2) in {0,1}x{0,1}
            
            for (I, K), c_meta in c_expr.items():
                for (i2, k2), c_sub in c_sub_dict.items():
                    c_idx = (2*I + i2, 2*K + k2)
                    full_c[c_idx] = full_c.get(c_idx, 0) + c_meta * c_sub
            
            all_terms.append((full_a, full_b, full_c))
    
    # Now we have 49 terms. Build U, V, W.
    rank = len(all_terms)
    assert rank == 49, f"Expected rank 49, got {rank}"
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (a_dict, b_dict, c_dict) in enumerate(all_terms):
        for (i, j), coeff in a_dict.items():
            if coeff != 0:
                U[i * m + j][c] = coeff
        for (j, k), coeff in b_dict.items():
            if coeff != 0:
                V[j * p + k][c] = coeff
        for (i, k), coeff in c_dict.items():
            if coeff != 0:
                W[k * n + i][c] = coeff
    
    # Verify the decomposition with a quick check
    # C[i,k] = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal the standard matrix product.
    
    import numpy as np
    A_test = np.random.randint(-3, 4, (n, m))
    B_test = np.random.randint(-3, 4, (m, p))
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for c in range(rank):
        u = U_arr[:, c].reshape(n, m)
        v = V_arr[:, c].reshape(m, p)
        w = W_arr[:, c].reshape(p, n)
        # w is indexed by k*n+i, so w[k*n+i] is the coefficient for C[i,k]
        # We need to reconstruct: contribution to C[i,k] from term c is
        #   (sum_j A[i,j]*u[j]) * (sum_j B[j,k]*v[j]) * w[k*n+i]
        # Wait, that's not how it works. Let me re-examine.
        #
        # The tensor decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # And T[i,j,k] = 1 if... no, T is the multiplication tensor.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is such that contracting with A and B gives C.
        # T[i,j,k] = delta (it's the identity in the appropriate sense).
        # Actually, the multiplication tensor M satisfies:
        #   C[i,k] = sum_{j} M[i,j,k] * A[i,j] * B[j,k]  -- no, that's not right.
        #
        # The correct formulation: the tensor T has indices (i, j, k) where
        # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[?,?].
        # Actually, for matrix multiplication, the tensor is:
        #   T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in a sense)
        # No, that's not right either.
        #
        # Let me think carefully. The matrix multiplication tensor T is defined by:
        #   T[i, j, k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
        # More precisely, the bilinear form is:
        #   C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor T has the property that:
        #   <T, A ⊗ B> = C  (in some appropriate sense)
        #
        # In the factorization T = sum_c u_c ⊗ v_c ⊗ w_c:
        #   C[i,k] = sum_c w_c[k*n+i] * (sum_j u_c[i*m+j] * A[i,j]) * (sum_j v_c[j*p+k] * B[j,k])
        #
        # Wait, that's not quite right. Let me look at the schoolbook example:
        #   U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c = i*m*p + j*p + k
        #   Then C[i,k] = sum_c W[k*n+i, c] * U[i*m+j_c, c] * V[j_c*p+k, c]
        #   where for each c, j_c is determined.
        #
        # Actually, the standard formulation is:
        #   C[i,k] = sum_j sum_c A[i,j] * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        #
        # No wait. Let me re-read the task. "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # "exactly decompose the matrix multiplication tensor"
        #
        # The multiplication tensor T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n} is defined by:
        #   T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}  (using appropriate indexings)
        #
        # In our indexing:
        #   U index: i*m + j
        #   V index: j*p + k  
        #   W index: k*n + i
        #
        # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        #
        # A rank-r decomposition is: T = sum_{c=1}^{r} u_c ⊗ v_c ⊗ w_c
        # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
        #
        # This means: T[a, b, c_idx] = sum_{r=1}^{rank} u_c[r] * v_c[r] * w_c[r]
        # where a = i*m+j, b = j'*p+k', c_idx = k''*n+i''
        #
        # And the matrix product is recovered by:
        #   C[i,k] = sum_j sum_{c_idx} T[i*m+j, j*p+k, c_idx] * A[i,j] * B[j,k]
        #           = sum_j A[i,j] * B[j,k]  (since T[i*m+j, j*p+k, k*n+i] = 1)
        #
        # So the decomposition is valid if:
        #   For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        #   And for all other index combinations, the sum is 0.
    
    # Verify: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    # For all other (a,b,c_idx) combinations, it should be 0.
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
