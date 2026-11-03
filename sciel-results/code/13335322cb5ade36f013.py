"""Matrix multiplication 6x6x6 using 2x2 block decomposition with rank-19 3x3 (Laderman)."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 6, 6, 6
    
    # Use Laderman's rank-19 algorithm for 3x3 matrix multiplication.
    # Then use Strassen at the 2x2 block level: total rank = 7 * 19 = 133.
    
    # Laderman's 19-term decomposition for 3x3:
    # Each term c has U33[i*3+j, c], V33[j*3+k, c], W33[k*3+i, c]
    # such that sum_c U33[i*3+j,c] * V33[j*3+k,c] * W33[k*3+i,c] = sum_j A[i,j]*B[j,k]
    
    # The 19 bilinear forms (products) and their contributions:
    # We define U33, V33, W33 directly as 9x19 matrices.
    
    r3 = 19
    U33 = [[0] * r3 for _ in range(9)]
    V33 = [[0] * r3 for _ in range(9)]
    W33 = [[0] * r3 for _ in range(9)]
    
    # Laderman's algorithm - 19 products:
    # Corrected W indices for singletons
    # m0  = a00 * b00 -> C00
    U33[0][0] = 1; V33[0][0] = 1; W33[0][0] = 1
    # m1  = a11 * b11 -> C11
    U33[4][1] = 1; V33[4][1] = 1; W33[4][1] = 1
    # m2  = a22 * b22 -> C22
    U33[8][2] = 1; V33[8][2] = 1; W33[8][2] = 1
    # m3  = a01 * b10 -> C00
    U33[1][3] = 1; V33[3][3] = 1; W33[0][3] = 1
    # m4  = a12 * b20 -> C10
    U33[5][4] = 1; V33[6][4] = 1; W33[3][4] = 1
    # m5  = a20 * b01 -> C21
    U33[6][5] = 1; V33[1][5] = 1; W33[5][5] = 1
    # m6  = a02 * b21 -> C02
    U33[2][6] = 1; V33[7][6] = 1; W33[6][6] = 1
    # m7  = a10 * b02 -> C12
    U33[3][7] = 1; V33[2][7] = 1; W33[7][7] = 1
    # m8  = a21 * b12 -> C22
    U33[7][8] = 1; V33[5][8] = 1; W33[8][8] = 1
    # m9  = (a00+a10) * (b00+b10)
    U33[0][9] = 1; U33[3][9] = 1
    V33[0][9] = 1; V33[3][9] = 1
    W33[0][9] = 1; W33[3][9] = 1
    # m10 = (a01+a11) * (b10+b11)
    U33[1][10] = 1; U33[4][10] = 1
    V33[3][10] = 1; V33[4][10] = 1
    W33[3][10] = 1; W33[4][10] = 1
    # m11 = (a02+a12) * (b20+b21)
    U33[2][11] = 1; U33[5][11] = 1
    V33[6][11] = 1; V33[7][11] = 1
    W33[6][11] = 1; W33[7][11] = 1
    # m12 = (a00+a20) * (b00+b20)
    U33[0][12] = 1; U33[6][12] = 1
    V33[0][12] = 1; V33[6][12] = 1
    W33[0][12] = 1; W33[6][12] = 1
    # m13 = (a01+a21) * (b10+b20)
    U33[1][13] = 1; U33[7][13] = 1
    V33[3][13] = 1; V33[6][13] = 1
    W33[3][13] = 1; W33[6][13] = 1
    # m14 = (a02+a22) * (b20+b21)
    U33[2][14] = 1; U33[8][14] = 1
    V33[6][14] = 1; V33[7][14] = 1
    W33[6][14] = 1; W33[7][14] = 1
    # m15 = (a10+a20) * (b01+b21)
    U33[3][15] = 1; U33[6][15] = 1
    V33[1][15] = 1; V33[7][15] = 1
    W33[1][15] = 1; W33[7][15] = 1
    # m16 = (a11+a21) * (b11+b21)
    U33[4][16] = 1; U33[7][16] = 1
    V33[4][16] = 1; V33[7][16] = 1
    W33[4][16] = 1; W33[7][16] = 1
    # m17 = (a12+a22) * (b21+b22)
    U33[5][17] = 1; U33[8][17] = 1
    V33[7][17] = 1; V33[8][17] = 1
    W33[7][17] = 1; W33[8][17] = 1
    # m18 = (a00+a01+a02) * (b00+b10+b20)
    U33[0][18] = 1; U33[1][18] = 1; U33[2][18] = 1
    V33[0][18] = 1; V33[3][18] = 1; V33[6][18] = 1
    W33[0][18] = 1; W33[3][18] = 1; W33[6][18] = 1
    
    # Now we need to verify this is correct. The reconstruction formulas for C[i,k]:
    # C00 = m0 - m9 + m12 + m18
    # C01 = m5 - m10 + m13 + m18  (need to verify)
    # ... this is getting complex. Let me just verify numerically.
    
    # Actually, let me just verify the tensor equation directly.
    # For each (i,j,k) in 0..2, check that:
    # sum_c U33[i*3+j,c] * V33[j*3+k,c] * W33[k*3+i,c] == (1 if j is between i and k appropriately)
    # Actually, the tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's the matmul tensor).
    
    # Let me verify:
    valid_laderman = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for c in range(r3):
                    s += U33[i*3+j][c] * V33[j*3+k][c] * W33[k*3+i][c]
                if s != 1:
                    valid_laderman = False
                    break
            if not valid_laderman: break
        if not valid_laderman: break
    
    if not valid_laderman:
        # Fall back to schoolbook 3x3
        r3 = 27
        U33 = [[0] * r3 for _ in range(9)]
        V33 = [[0] * r3 for _ in range(9)]
        W33 = [[0] * r3 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U33[i*3+j][col] = 1
                    V33[j*3+k][col] = 1
                    W33[k*3+i][col] = 1
                    col += 1
    
    # Since I'm not 100% sure of Laderman's exact coefficients, let me use a verified approach.
    # I'll use the schoolbook for 3x3 (rank 27) combined with Strassen (7 multiplications) = 189.
    # That's what we already have. Let me try a different approach.
    
    # Actually, let me just use schoolbook 3x3 with Strassen. Rank = 7*27 = 189.
    r3 = 27
    U33 = [[0] * r3 for _ in range(9)]
    V33 = [[0] * r3 for _ in range(9)]
    W33 = [[0] * r3 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U33[i*3+j][col] = 1
                V33[j*3+k][col] = 1
                W33[k*3+i][col] = 1
                col += 1
    
    # Now build the full 6x6 decomposition using Strassen at the block level.
    # Total rank = 7 * r3 = 7 * 27 = 189
    
    rank = 7 * r3
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    strassen_terms = [
        {'a': {'A00': 1, 'A11': 1}, 'b': {'B00': 1, 'B11': 1}, 'c': {'C00': 1, 'C11': 1}},
        {'a': {'A10': 1, 'A11': 1}, 'b': {'B00': 1}, 'c': {'C10': 1, 'C11': -1}},
        {'a': {'A00': 1}, 'b': {'B01': 1, 'B11': -1}, 'c': {'C01': 1, 'C11': 1}},
        {'a': {'A11': 1}, 'b': {'B10': 1, 'B00': -1}, 'c': {'C00': 1, 'C10': 1}},
        {'a': {'A00': 1, 'A01': 1}, 'b': {'B11': 1}, 'c': {'C00': -1, 'C01': 1}},
        {'a': {'A10': 1, 'A00': -1}, 'b': {'B00': 1, 'B01': 1}, 'c': {'C11': 1}},
        {'a': {'A01': 1, 'A11': -1}, 'b': {'B10': 1, 'B11': 1}, 'c': {'C00': 1}},
    ]
    
    block_map = {
        'A00': (0, 0), 'A01': (0, 1), 'A10': (1, 0), 'A11': (1, 1),
        'B00': (0, 0), 'B01': (0, 1), 'B10': (1, 0), 'B11': (1, 1),
        'C00': (0, 0), 'C01': (0, 1), 'C10': (1, 0), 'C11': (1, 1),
    }
    
    for m_idx, term in enumerate(strassen_terms):
        base_col = m_idx * r3
        for local_col in range(r3):
            col = base_col + local_col
            i3 = local_col // 9
            rem = local_col % 9
            j3 = rem // 3
            k3 = rem % 3
            for a_block, ca in term['a'].items():
                bi, bj = block_map[a_block]
                u_row = (bi*3+i3)*6 + (bj*3+j3)
                U[u_row][col] += ca * U33[i3*3+j3][local_col]
            for b_block, cb in term['b'].items():
                bk, bl = block_map[b_block]
                v_row = (bk*3+j3)*6 + (bl*3+k3)
                V[v_row][col] += cb * V33[j3*3+k3][local_col]
            for c_block, cc in term['c'].items():
                ci, ck = block_map[c_block]
                w_row = (ck*3+k3)*6 + (ci*3+i3)
                W[w_row][col] += cc * W33[k3*3+i3][local_col]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
