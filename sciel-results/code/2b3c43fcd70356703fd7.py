"""Matrix multiplication 3x4x4 via Strassen block decomposition (rank 32)."""
import time

DIMENSIONS = [3, 4, 4]
_DEADLINE = time.time() + 140.0

# EVOLVE-BLOCK-START
def construct():
    # 3x4 * 4x4 via 2x2 Strassen blocks.
    # A = [A1 (2x2); a3 (1x4)], B = [B1 B2; B3 B4] (each 2x2).
    # C = A*B: C[:2,:] = A1*B1 + a3*B3 (2x2 via Strassen) + a3*B4 (1x2 schoolbook)
    #        C[2,:]  = a3*B3 + a3*B4 (1x4 schoolbook)
    # Rank = 4*7 (Strassen for A1*B1 and a3*B3) + 4 (schoolbook for a3*B4) = 32.

    # Strassen 2x2x2 decomposition.
    # Inputs: A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]].
    # M1=(A00+A11)(B00+B11)
    # M2=(A10+A11)B00
    # M3=A00(B10-B00)
    # M4=A11(B11-A00)
    # M5=(A00+A01)B11
    # M6=(A10-A00)(B00+B01)
    # M7=(A01-A11)(B10+B11)
    # C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
    def strassen(A_idx, B_idx):
        # A_idx: [a00,a01,a10,a11], B_idx: [b00,b01,b10,b11]
        a00, a01, a10, a11 = A_idx
        b00, b01, b10, b11 = B_idx
        terms = []
        # M1: (a00+a11) * (b00+b11)
        terms.append(([a00, a11], [b00, b11]))
        # M2: (a10+a11) * b00
        terms.append(([a10, a11], [b00]))
        # M3: a00 * (b10-b00)
        terms.append(([a00], [b10, -b00]))
        # M4: a11 * (b11-a00)
        terms.append(([a11], [b11, -a00]))
        # M5: (a00+a01) * b11
        terms.append(([a00, a01], [b11]))
        # M6: (a10-a00) * (b00+b01)
        terms.append(([a10, -a00], [b00, b01]))
        # M7: (a01-a11) * (b10+b11)
        terms.append(([a01, -a11], [b10, b11]))
        return terms

    # A1 indices (rows 0,1 of A) x (cols 0,1 of B): a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
    # B1 indices: b00=B[0,0], b01=B[0,1], b10=B[1,0], b11=B[1,1]
    A1 = [0, 1, 4, 5]          # A[0,0], A[0,1], A[1,0], A[1,1]
    B1 = [0, 1, 4, 5]          # B[0,0], B[0,1], B[1,0], B[1,1]
    # a3 = A[2,0..3] -> A indices [8,9,10,11]
    a3 = [8, 9, 10, 11]
    # B3 = B[2,0..3] -> B indices [8,9,10,11]
    B3 = [8, 9, 10, 11]
    # B4 = B[2,0..3]... wait, B is 4x4. Let me re-index.

    # Re-index B properly. B is 4x4: B[r,c] index = r*4+c.
    # B1 = top-left 2x2: B[0,0],B[0,1],B[1,0],B[1,1] = [0,1,4,5]
    # B2 = top-right 2x2: B[0,2],B[0,3],B[1,2],B[1,3] = [2,3,6,7]
    # B3 = bottom-left 2x2: B[2,0],B[2,1],B[3,0],B[3,1] = [8,9,12,13]
    # B4 = bottom-right 2x2: B[2,2],B[2,3],B[3,2],B[3,3] = [10,11,14,15]
    B2 = [2, 3, 6, 7]
    B3 = [8, 9, 12, 13]
    B4 = [10, 11, 14, 15]

    # A is 3x4: A[r,c] index = r*4+c.
    # A1 = top 2x2: A[0,0],A[0,1],A[1,0],A[1,1] = [0,1,4,5]
    # a3 = A[2,0..3] = [8,9,10,11]
    A1 = [0, 1, 4, 5]
    a3 = [8, 9, 10, 11]

    # Build all terms: each term is (U_list, V_list, W_list) where
    # U_list = [(A_index, coeff), ...], V_list = [(B_index, coeff), ...],
    # W_list = [(C_index, coeff), ...]
    all_terms = []

    def add_strassen(A_idx, B_idx, C_offset):
        # C_offset: function mapping (r,c) in 2x2 to C index
        terms = strassen(A_idx, B_idx)
        # M outputs: C00=M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
        # M1->C00(+1),C11(+1); M2->C10(+1),C11(-1); M3->C01(+1),C11(+1)
        # M4->C00(+1),C10(+1); M5->C00(-1),C01(+1); M6->C11(+1); M7->C00(+1)
        # We need to distribute each Mx to the C positions with their coefficients.
        # M1: (U1,V1) contributes to C00(+1) and C11(+1) -> two rank-1 terms
        # Actually each Mx is ONE rank-1 term (one U combo, one V combo).
        # It contributes to multiple C positions with different coefficients.
        # So Mx produces ONE column in U,V and a W vector with multiple nonzero entries.
        # M1: U=(a00,a11), V=(b00,b11), W: C00+1, C11+1
        # M2: U=(a10,a11), V=(b00), W: C10+1, C11-1
        # M3: U=(a00), V=(b10,-b00), W: C01+1, C11+1
        # M4: U=(a11), V=(b11,-a00), W: C00+1, C10+1
        # M5: U=(a00,a01), V=(b11), W: C00-1, C01+1
        # M6: U=(a10,-a00), V=(b00,b01), W: C11+1
        # M7: U=(a01,-a11), V=(b10,b11), W: C00+1
        M1 = terms[0]
        all_terms.append((M1[0], M1[1], [(C_offset(0,0), 1), (C_offset(1,1), 1)]))
        M2 = terms[1]
        all_terms.append((M2[0], M2[1], [(C_offset(1,0), 1), (C_offset(1,1), -1)]))
        M3 = terms[2]
        all_terms.append((M3[0], M3[1], [(C_offset(0,1), 1), (C_offset(1,1), 1)]))
        M4 = terms[3]
        all_terms.append((M4[0], M4[1], [(C_offset(0,0), 1), (C_offset(1,0), 1)]))
        M5 = terms[4]
        all_terms.append((M5[0], M5[1], [(C_offset(0,0), -1), (C_offset(0,1), 1)]))
        M6 = terms[5]
        all_terms.append((M6[0], M6[1], [(C_offset(1,1), 1)]))
        M7 = terms[6]
        all_terms.append((M7[0], M7[1], [(C_offset(0,0), 1)]))

    # C is 3x4: C[r,c] index = r*4+c.
    # A1*B1 -> C[0:2, 0:2], offset (r,c) -> r*4+c
    def off_AB1(r, c):
        return r * 4 + c
    add_strassen(A1, B1, off_AB1)

    # a3*B3 -> C[2, 0:2]. a3 is 1x4, B3 is 2x2 (rows 2,3 of B, cols 0,1).
    # C[2,c] = sum_{k=0..3} a3[k]*B[2+k, c] for c in 0,1.
    # This is (a3[0],a3[1],a3[2],a3[3]) dot (B3[0,c],B3[1,c],B3[2,c],B3[3,c]).
    # B3 indices: row0col0=B[2,0]=8, row0col1=B[2,1]=9, row1col0=B[3,0]=12, row1col1=B[3,1]=13
    # For c=0: B3 cols = [8,12] (B[2,0],B[3,0])
    # For c=1: B3 cols = [9,13] (B[2,1],B[3,1])
    # Wait, a3*B3: a3 is 1x4, B3 is 4x2? No.
    # Let me reconsider. a3 = A[2,:] is 1x4. B3 = B[2:4, 0:2] is 2x2.
    # a3 * B3 doesn't dimension-match directly. Let me re-examine.
    # A is 3x4, B is 4x4. C = A*B is 3x4.
    # C[2, c] = sum_{k=0}^3 A[2,k]*B[k,c].
    # Split B into B3 (rows 0,1) and B4 (rows 2,3)? No.
    # Let me partition differently. B rows 0,1 = top, rows 2,3 = bottom.
    # C[2,c] = sum_{k=0,1} A[2,k]*B[k,c] + sum_{k=2,3} A[2,k]*B[k,c]
    #        = a3_top * B_top[c] + a3_bottom * B_bottom[c]
    # where a3_top = A[2,0:2] (1x2), a3_bottom = A[2,2:4] (1x2)
    # B_top[c] = B[0:2, c] (2x1), B_bottom[c] = B[2:4, c] (2x1)
    # For c in 0..3:
    # C[2,c] = A[2,0]*B[0,c] + A[2,1]*B[1,c] + A[2,2]*B[2,c] + A[2,3]*B[3,c]
    # This is just 4 schoolbook terms for each c, total 16 terms. That's too many.
    # Let me reconsider the whole partitioning.

    # Better approach: partition A into A1 (2x2) and a3 (1x4).
    # Partition B into B1 (2x2 top-left), B2 (2x2 top-right), B3 (2x2 bottom-left), B4 (2x2 bottom-right).
    # C = A*B.
    # C[0:2, 0:2] = A1*B1 (2x2 Strassen, 7 terms)
    # C[0:2, 2:4] = A1*B2 (2x2 Strassen, 7 terms)
    # C[2, 0:2] = a3*B3 (1x4 * 2x2 -> 1x2, schoolbook 4 terms)
    # C[2, 2:4] = a3*B4 (1x4 * 2x2 -> 1x2, schoolbook 4 terms)
    # Total = 7+7+4+4 = 22. But wait, a3*B3: a3 is 1x4, B3 is 2x2.
    # a3*B3 doesn't make sense dimensionally. Let me re-examine.
    # A is 3x4, B is 4x4.
    # A1 = A[0:2, 0:2] (2x2), a3 = A[2:3, :] (1x4)
    # B1 = B[0:2, 0:2] (2x2), B2 = B[0:2, 2:4] (2x2)
    # B3 = B[2:4, 0:2] (2x2), B4 = B[2:4, 2:4] (2x2)
    # C[0:2, 0:2] = A1*B1 (2x2*2x2 = 2x2) -> Strassen 7 terms
    # C[0:2, 2:4] = A1*B2 (2x2*2x2 = 2x2) -> Strassen 7 terms
    # C[2:3, 0:2] = a3*B3 (1x4 * 4x2? No, B3 is 2x2)
    # a3 is 1x4, B3 is B[2:4, 0:2] which is 2x2. a3*B3 is 1x4 * 2x2 = invalid.
    # I think I need B3 to be 4x2. Let me re-partition.
    # B is 4x4. Partition into top 2 rows and bottom 2 rows.
    # B_top = B[0:2, :] (2x4), B_bottom = B[2:4, :] (2x4)
    # C[0:2, :] = A1 * B_top (2x2 * 2x4 = 2x4)
    # C[2:3, :] = a3 * B (1x4 * 4x4 = 1x4)
    # C[0:2, 0:2] = A1 * B_top[:, 0:2] = A1 * B1 (2x2*2x2) -> Strassen 7
    # C[0:2, 2:4] = A1 * B_top[:, 2:4] = A1 * B2 (2x2*2x2) -> Strassen 7
    # C[2, 0:2] = a3 * B[:, 0:2] (1x4 * 4x2 = 1x2) -> schoolbook 4 terms
    # C[2, 2:4] = a3 * B[:, 2:4] (1x4 * 4x2 = 1x2) -> schoolbook 4 terms
    # Total = 7+7+4+4 = 22.

    # Let me implement this correctly.
    all_terms = []

    # Strassen for A1*B1 -> C[0:2, 0:2]
    A1 = [0, 1, 4, 5]  # A[0,0],A[0,1],A[1,0],A[1,1]
    B1 = [0, 1, 4, 5]  # B[0,0],B[0,1],B[1,0],B[1,1]
    B2 = [2, 3, 6, 7]  # B[0,2],B[0,3],B[1,2],B[1,3]

    def strassen_terms(A_idx, B_idx, C_offset):
        a00, a01, a10, a11 = A_idx
        b00, b01, b10, b11 = B_idx
        terms = []
        # M1: (a00+a11)*(b00+b11) -> C00+1, C11+1
        terms.append(([a00, a11], [b00, b11], [(C_offset(0,0), 1), (C_offset(1,1), 1)]))
        # M2: (a10+a11)*b00 -> C10+1, C11-1
        terms.append(([a10, a11], [b00], [(C_offset(1,0), 1), (C_offset(1,1), -1)]))
        # M3: a00*(b10-b00) -> C01+1, C11+1
        terms.append(([a00], [b10, -b00], [(C_offset(0,1), 1), (C_offset(1,1), 1)]))
        # M4: a11*(b11-a00) -> C00+1, C10+1
        terms.append(([a11], [b11, -a00], [(C_offset(0,0), 1), (C_offset(1,0), 1)]))
        # M5: (a00+a01)*b11 -> C00-1, C01+1
        terms.append(([a00, a01], [b11], [(C_offset(0,0), -1), (C_offset(0,1), 1)]))
        # M6: (a10-a00)*(b00+b01) -> C11+1
        terms.append(([a10, -a00], [b00, b01], [(C_offset(1,1), 1)]))
        # M7: (a01-a11)*(b10+b11) -> C00+1
        terms.append(([a01, -a11], [b10, b11], [(C_offset(0,0), 1)]))
        return terms

    # A1*B1 -> C[0:2, 0:2], offset (r,c) = r*4+c
    all_terms.extend(strassen_terms(A1, B1, lambda r, c: r * 4 + c))

    # A1*B2 -> C[0:2, 2:4], offset (r,c) = r*4 + (c+2)
    all_terms.extend(strassen_terms(A1, B2, lambda r, c: r * 4 + (c + 2)))

    # a3*B[:, 0:2] -> C[2, 0:2], schoolbook 4 terms
    # C[2,c] = sum_{k=0}^3 A[2,k]*B[k,c] for c in 0,1
    # B[:, 0:2] indices: B[0,0]=0, B[1,0]=4, B[2,0]=8, B[3,0]=12 (for c=0)
    #                    B[0,1]=1, B[1,1]=5, B[2,1]=9, B[3,1]=13 (for c=1)
    a3 = [8, 9, 10, 11]  # A[2,0],A[2,1],A[2,2],A[2,3]
    for c in range(2):
        B_col = [c, 4 + c, 8 + c, 12 + c]  # B[0,c],B[1,c],B[2,c],B[3,c]
        for k in range(4):
            all_terms.append(([a3[k]], [B_col[k]], [(2 * 4 + c, 1)]))

    # a3*B[:, 2:4] -> C[2, 2:4], schoolbook 4 terms
    for c in range(2, 4):
        B_col = [c, 4 + c, 8 + c, 12 + c]  # B[0,c],B[1,c],B[2,c],B[3,c]
        for k in range(4):
            all_terms.append(([a3[k]], [B_col[k]], [(2 * 4 + c, 1)]))

    rank = len(all_terms)
    n, m, p = DIMENSIONS  # 3, 4, 4
    U = [[0] * rank for _ in range(n * m)]  # 12 x rank
    V = [[0] * rank for _ in range(m * p)]  # 16 x rank
    W = [[0] * rank for _ in range(p * n)]  # 12 x rank

    for col, (U_list, V_list, W_list) in enumerate(all_terms):
        for (idx, coeff) in U_list:
            U[idx][col] += coeff
        for (idx, coeff) in V_list:
            V[idx][col] += coeff
        for (idx, coeff) in W_list:
            W[idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
