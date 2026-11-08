"""Rewritten program: rank-189 decomposition of 6x6 matrix multiplication.
Uses Kronecker product of Strassen 2x2 (rank 7) and schoolbook 3x3 (rank 27).
Index convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=6.
"""

def construct():
    n, m, p = 6, 6, 6

    # Strassen 2x2 decomposition (rank 7)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # U factors for A (2x2): A is indexed by (row, col) in {0,1}x{0,1}
    # A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # U_2x2[i, t] where i is the linear index in A (0..3), t is term 0..6
    U2 = [
        [1, 0, 1, 0, 1, 1, 0],  # A00: M1, M3, M5, M6
        [0, 0, 0, 0, 1, 0, 1],  # A01: M5, M7
        [0, 1, 0, 0, 0, -1, 1], # A10: M2, M6, M7
        [1, 1, 0, 1, 0, 0, -1], # A11: M1, M2, M4, M7
    ]

    # V factors for B (2x2): B indexed by (row, col) in {0,1}x{0,1}
    # B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],  # B00: M1, M2, M6
        [0, 0, 0, 1, 0, 0, 1],  # B01: M4, M7
        [0, 0, 1, -1, 0, 1, 0], # B10: M3, M4, M6
        [1, 0, -1, 0, 1, 0, 0], # B11: M1, M3, M5
    ]

    # W factors for C (2x2): C indexed by (row, col) in {0,1}x{0,1}
    # C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C00: M1, M4, M5, M7
        [0, 0, 1, 0, 1, 0, 0],    # C01: M3, M5
        [0, 1, 0, 1, 0, 0, 0],    # C10: M2, M4
        [1, -1, 1, 0, 0, 1, 0],   # C11: M1, M2, M3, M6
    ]

    # Schoolbook 3x3 decomposition (rank 27)
    # U3[i, t] where i is linear index in A (0..8), t is term 0..26
    # Term t corresponds to (i3, j3, k3) with t = i3*9 + j3*3 + k3
    # U3[i3*3+j3, i3*9+j3*3+k3] = 1 if matching
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                t = i3*9 + j3*3 + k3
                U3[i3*3+j3][t] = 1
                V3[j3*3+k3][t] = 1
                W3[k3*3+i3][t] = 1

    # Build 6x6 factors via Kronecker product
    # 6 = 2*3, so we split indices: outer (2-part) and inner (3-part)
    # A index: i6 = i2*3 + i3, where i2 in {0,1} (block row), i3 in {0,1,2} (within block)
    # Wait, for Kronecker product of matrix multiplication:
    # We view 6x6 matrices as 2x2 blocks of 3x3 matrices.
    # A = [[A00, A01], [A10, A11]] where each Aij is 3x3
    # The multiplication is block-wise: C = A*B
    # Each Cij = sum_k Akj * Bjk (3x3 matrix multiplication)
    # So we need Strassen on the 2x2 block structure, and within each block, schoolbook 3x3.
    
    # Actually, the correct Kronecker decomposition for block matrix multiplication:
    # If we have 2x2 blocks of size 3x3, the overall tensor rank is rank_strassen_2x2 * rank_schoolbook_3x3 = 7*27 = 189.
    # 
    # Index mapping:
    # A has 6x6 = 36 entries. Block (a2, b2) contains 3x3 = 9 entries indexed by (a3, b3).
    # Linear index in A: idx_A = (a2*3 + a3)*6 + (b2*3 + b3) where a2,b2 in {0,1}, a3,b3 in {0,1,2}
    #   = a2*18 + a3*6 + b2*3 + b3
    #   = (a2*3 + a3)*6 + (b2*3 + b3)
    #
    # For the Strassen part (2x2 blocks):
    #   Block index: (a2, b2) -> linear block index = a2*2 + b2
    #   U2 is indexed by [block_idx_A, strassen_term]
    #   V2 is indexed by [block_idx_B, strassen_term]
    #   W2 is indexed by [block_idx_C, strassen_term]
    #
    # For the schoolbook part (3x3 within blocks):
    #   Within-block index: (a3, b3) for A, (b3, c3) for B, (c3, a3) for C
    #   U3 is indexed by [(a3*3+b3), schoolbook_term]
    #   V3 is indexed by [(b3*3+c3), schoolbook_term]
    #   W3 is indexed by [(c3*3+a3), schoolbook_term]
    #
    # Combined:
    #   U6[idx_A, combined_term] = U2[a2*2+b2, s_term] * U3[a3*3+b3, t_term]
    #   where combined_term = s_term * 27 + t_term
    #
    #   V6[idx_B, combined_term] = V2[b2*2+c2, s_term] * V3[b3*3+c3, t_term]
    #
    #   W6[idx_C, combined_term] = W2[c2*2+a2, s_term] * W3[c3*3+a3, t_term]
    #
    #   idx_A = (a2*3+a3)*6 + (b2*3+b3)
    #   idx_B = (b2*3+b3)*6 + (c2*3+c3)
    #   idx_C = (c2*3+c3)*6 + (a2*3+a3)  -- W convention: W[k*n+i, c] where k is row in C's "col" part, i is row in C's "row" part
    #
    # Wait, the W convention: W[k*n+i, r] where k corresponds to the column index of C, and i to the row index of C.
    # C is 6x6. C[row, col]. row = c2*3+c3, col = a2*3+a3.
    # W index = col*n + row = (a2*3+a3)*6 + (c2*3+c3)
    #
    # Let me re-derive:
    # U: (n*m, rank) = (36, 189). Index = i*m+j = i*6+j, where i is row of A, j is col of A.
    #   i = a2*3+a3, j = b2*3+b3. So U index = (a2*3+a3)*6 + (b2*3+b3).
    # V: (m*p, rank) = (36, 189). Index = j*p+k = j*6+k, where j is row of B, k is col of B.
    #   j = b2*3+b3, k = c2*3+c3. So V index = (b2*3+b3)*6 + (c2*3+c3).
    # W: (p*n, rank) = (36, 189). Index = k*n+i = k*6+i, where k is col of C, i is row of C.
    #   k = a2*3+a3, i = c2*3+c3. So W index = (a2*3+a3)*6 + (c2*3+c3).

    rank = 7 * 27
    U6 = [[0]*rank for _ in range(36)]
    V6 = [[0]*rank for _ in range(36)]
    W6 = [[0]*rank for _ in range(36)]

    for s_term in range(7):
        for t_term in range(27):
            ct = s_term * 27 + t_term
            for a2 in range(2):
                for b2 in range(2):
                    for a3 in range(3):
                        for b3 in range(3):
                            u2_val = U2[a2*2+b2][s_term]
                            u3_val = U3[a3*3+b3][t_term]
                            if u2_val != 0 and u3_val != 0:
                                idx_A = (a2*3+a3)*6 + (b2*3+b3)
                                U6[idx_A][ct] += u2_val * u3_val

            for b2 in range(2):
                for c2 in range(2):
                    for b3 in range(3):
                        for c3 in range(3):
                            v2_val = V2[b2*2+c2][s_term]
                            v3_val = V3[b3*3+c3][t_term]
                            if v2_val != 0 and v3_val != 0:
                                idx_B = (b2*3+b3)*6 + (c2*3+c3)
                                V6[idx_B][ct] += v2_val * v3_val

            for c2 in range(2):
                for a2 in range(2):
                    for c3 in range(3):
                        for a3 in range(3):
                            w2_val = W2[c2*2+a2][s_term]
                            w3_val = W3[c3*3+a3][t_term]
                            if w2_val != 0 and w3_val != 0:
                                idx_W = (a2*3+a3)*6 + (c2*3+c3)
                                W6[idx_W][ct] += w2_val * w3_val

    return {"U": U6, "V": V6, "W": W6}
