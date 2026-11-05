"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructive rank-50 decomposition of the 2x4x7 matrix multiplication tensor.
    
    Strategy: Decompose the 2x4x7 multiplication into two 2x2x7 multiplications.
    Each 2x2x7 is decomposed as three 2x2x2 (Strassen, rank 7 each) plus one 2x2x1 (schoolbook, rank 4).
    Total: 2 * (3*7 + 4) = 2 * 25 = 50.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Strassen coefficients for 2x2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    #
    # Represent each rank-1 term as (u, v, w) where:
    # u is a 2x2 matrix (flattened row-major: [u00, u01, u10, u11])
    # v is a 2x2 matrix (flattened row-major: [v00, v01, v10, v11])
    # w is a 2x2 matrix (flattened row-major: [w00, w01, w10, w11])
    
    def strassen_terms():
        """Return 7 rank-1 terms for 2x2x2 multiplication."""
        terms = []
        # M1 = (a11+a22)(b11+b22)
        u = [1, 0, 1, 0]   # a11+a22
        v = [1, 0, 1, 0]   # b11+b22
        w = [1, 0, 0, 1]   # contributes to c11 and c22
        terms.append((u, v, w))
        
        # M2 = (a21+a22)b11
        u = [0, 0, 1, 1]   # a21+a22
        v = [1, 0, 0, 0]   # b11
        w = [0, 0, 1, 0]   # contributes to c21 (and c22 with -1)
        terms.append((u, v, w))
        
        # M3 = a11(b21-b22)
        u = [1, 0, 0, 0]   # a11
        v = [0, 0, 1, -1]  # b21-b22
        w = [0, 1, 0, 1]   # contributes to c12 and c22
        terms.append((u, v, w))
        
        # M4 = a22(b12-b11)
        u = [0, 0, 0, 1]   # a22
        v = [0, 1, 0, 0]   # b12
        w = [1, 0, 0, 0]   # contributes to c11 (wait, let me re-derive)
        # Actually: c11 = M1 + M4 - M5 + M7, c12 = M3 + M5, c21 = M2 + M4, c22 = M1 - M2 + M3 + M6
        # Let me redo this carefully.
        # Standard Strassen:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b22)
        # M4 = a22(b12-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        #
        # c11 = M1 - M5 + M4 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # I'll clear and redo properly
        return None
    
    # Let me just implement it directly by checking each term
    # For 2x2x2, the 7 terms are:
    def get_strassen_2x2x2():
        """
        Returns list of (u_flat, v_flat, w_flat) for 7 rank-1 terms.
        u_flat = [u00, u01, u10, u11] (row-major 2x2)
        v_flat = [v00, v01, v10, v11]
        w_flat = [w00, w01, w10, w11]
        """
        terms = []
        
        # M1 = (a11+a22)(b11+b22), contributes: c11 += 1, c22 += 1
        # u: a11+a22 => [1,0,1,0]... wait, row-major: row0=[a11,a12], row1=[a21,a22]
        # So a11+a22 as a linear form: coefficient of a11=1, a12=0, a21=0, a22=1 => [1,0,0,1]
        # b11+b22: coefficient of b11=1, b12=0, b21=0, b22=1 => [1,0,0,1]
        # w: c11 gets +1, c22 gets +1 => [1,0,0,1]
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        
        # M2 = (a21+a22)b11, contributes: c21 += 1, c22 += -1
        # u: a21+a22 => [0,0,1,1]
        # v: b11 => [1,0,0,0]
        # w: c21 += 1, c22 += -1 => [0,0,1,-1]
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
        
        # M3 = a11(b21-b22), contributes: c12 += 1, c22 += 1
        # u: a11 => [1,0,0,0]
        # v: b21-b22 => [0,0,1,-1]
        # w: c12 += 1, c22 += 1 => [0,1,0,1]
        terms.append(([1,0,0,0], [0,0,1,-1], [0,1,0,1]))
        
        # M4 = a22(b12-b11), contributes: c11 += 1, c21 += 1
        # u: a22 => [0,0,0,1]
        # v: b12-b11 => [-1,1,0,0]
        # w: c11 += 1, c21 += 1 => [1,0,1,0]
        terms.append(([0,0,0,1], [-1,1,0,0], [1,0,1,0]))
        
        # M5 = (a11+a12)b22, contributes: c11 += -1, c12 += 1
        # u: a11+a12 => [1,1,0,0]
        # v: b22 => [0,0,0,1]
        # w: c11 += -1, c12 += 1 => [-1,1,0,0]
        terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
        
        # M6 = (a21-a11)(b11+b12), contributes: c22 += 1
        # u: a21-a11 => [-1,0,1,0]
        # v: b11+b12 => [1,1,0,0]
        # w: c22 += 1 => [0,0,0,1]
        terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
        
        # M7 = (a12-a22)(b21+b22), contributes: c11 += 1
        # u: a12-a22 => [0,1,0,-1]
        # v: b21+b22 => [0,0,1,1]
        # w: c11 += 1 => [1,0,0,0]
        terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
        
        return terms
    
    def get_schoolbook_2x2x1():
        """
        Returns 4 rank-1 terms for 2x2x1 multiplication (2x2 matrix times 2x1 vector = 2x1 vector).
        But in our context, this is part of 2x2x7 where the last 'column block' is 2x1.
        Actually, for 2x2x7, we have columns 0-5 as three 2x2 blocks and column 6 as a 2x1 block.
        The 2x2x1 part: A (2x2) times B_col (2x1) = C_col (2x1).
        Rank 4 schoolbook.
        
        u_flat: 2x2 linear form on A
        v_flat: 2x1 linear form on B (only 2 entries: [v0, v1])
        w_flat: 2x1 linear form on C (only 2 entries: [w0, w1])
        """
        terms = []
        # c0 = a00*b0 + a01*b1
        terms.append(([1,0,0,0], [1,0], [1,0]))
        terms.append(([0,1,0,0], [0,1], [1,0]))
        # c1 = a10*b0 + a11*b1
        terms.append(([0,0,1,0], [1,0], [0,1]))
        terms.append(([0,0,0,1], [0,1], [0,1]))
        return terms
    
    def get_strassen_2x2x2_full():
        """
        Returns 7 rank-1 terms for 2x2x2.
        Each term: (u_flat_4, v_flat_4, w_flat_4) where each flat is length 4 (2x2 row-major).
        """
        return get_strassen_2x2x2()
    
    # Now build the 2x4x7 decomposition.
    # A is 2x4, B is 4x7, C is 2x7.
    # Split A into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3).
    # Split B into B_top (2x7, rows 0-1) and B_bottom (2x7, rows 2-3).
    # C = A_left * B_top + A_right * B_bottom.
    
    # For each 2x2x7 multiplication:
    # B (2x7) is split into B0 (2x2, cols 0-1), B1 (2x2, cols 2-3), B2 (2x2, cols 4-5), B3 (2x1, col 6).
    # C (2x7) = C0 (2x2) + C1 (2x2) + C2 (2x2) + C3 (2x1) where Ci = A * Bi.
    # Each 2x2x2 uses Strassen (rank 7), the 2x2x1 uses schoolbook (rank 4).
    # Total per 2x2x7: 3*7 + 4 = 25.
    
    # Index conventions:
    # U has shape (n*m, rank) = (2*4, rank) = (8, rank).
    #   U[i*m + j, r] where i in [0,1], j in [0,3].
    # V has shape (m*p, rank) = (4*7, rank) = (28, rank).
    #   V[j*p + k, r] where j in [0,3], k in [0,6].
    # W has shape (p*n, rank) = (7*2, rank) = (14, rank).
    #   W[k*n + i, r] where k in [0,6], i in [0,1].
    
    # For the left half (A_left, B_top):
    # A_left uses A columns 0,1 => i in [0,1], j in [0,1] => U indices: i*4 + j, so 0,1,4,5.
    # B_top uses B rows 0,1 => j in [0,1], k in [0,6] => V indices: j*7 + k, so 0..6, 7..13.
    # C columns all 0..6 => W indices: k*2 + i, so 0,1,2,3,4,5,6,7,8,9,10,11,12,13.
    
    # For the right half (A_right, B_bottom):
    # A_right uses A columns 2,3 => i in [0,1], j in [2,3] => U indices: i*4 + j, so 2,3,6,7.
    # B_bottom uses B rows 2,3 => j in [2,3], k in [0,6] => V indices: j*7 + k, so 14..20, 21..27.
    # C columns all 0..6 => W indices: k*2 + i, so 0,1,...,13 (same as left).
    
    strassen_terms = get_strassen_2x2x2_full()
    schoolbook_terms_2x2x1 = get_schoolbook_2x2x1()
    
    # Collect all terms
    all_u = []
    all_v = []
    all_w = []
    
    def add_term_left(u4, v4, w4, col_offset_b):
        """
        Add a term for the left half (A_left, B_top).
        u4: length 4, [u00, u01, u10, u11] for A_left (2x2)
        v4: length 4, [v00, v01, v10, v11] for B block (2x2) at columns col_offset_b, col_offset_b+1
        w4: length 4, [w00, w01, w10, w11] for C block (2x2) at columns col_offset_b, col_offset_b+1
        """
        u = [0] * 8
        v = [0] * 28
        w = [0] * 14
        
        # Map u4 to U indices: i*4+j for i in 0,1 and j in 0,1
        u[0] = u4[0]  # i=0,j=0
        u[1] = u4[1]  # i=0,j=1
        u[4] = u4[2]  # i=1,j=0
        u[5] = u4[3]  # i=1,j=1
        
        # Map v4 to V indices: j*7+k for j in 0,1 and k in col_offset_b, col_offset_b+1
        v[0*7 + col_offset_b] = v4[0]   # j=0, k=col_offset_b
        v[0*7 + col_offset_b+1] = v4[1] # j=0, k=col_offset_b+1
        v[1*7 + col_offset_b] = v4[2]   # j=1, k=col_offset_b
        v[1*7 + col_offset_b+1] = v4[3] # j=1, k=col_offset_b+1
        
        # Map w4 to W indices: k*2+i for k in col_offset_b, col_offset_b+1 and i in 0,1
        w[col_offset_b*2 + 0] = w4[0]   # k=col_offset_b, i=0
        w[col_offset_b*2 + 1] = w4[1]   # k=col_offset_b, i=1
        w[(col_offset_b+1)*2 + 0] = w4[2]   # k=col_offset_b+1, i=0
        w[(col_offset_b+1)*2 + 1] = w4[3]   # k=col_offset_b+1, i=1
        
        all_u.append(u)
        all_v.append(v)
        all_w.append(w)
    
    def add_term_right(u4, v4, w4, col_offset_b):
        """
        Add a term for the right half (A_right, B_bottom).
        u4: length 4, [u00, u01, u10, u11] for A_right (2x2)
        v4: length 4, [v00, v01, v10, v11] for B block (2x2) at columns col_offset_b, col_offset_b+1
        w4: length 4, [w00, w01, w10, w11] for C block (2x2) at columns col_offset_b, col_offset_b+1
        """
        u = [0] * 8
        v = [0] * 28
        w = [0] * 14
        
        # Map u4 to U indices: i*4+j for i in 0,1 and j in 2,3
        u[0*4 + 2] = u4[0]  # i=0,j=2
        u[0*4 + 3] = u4[1]  # i=0,j=3
        u[1*4 + 2] = u4[2]  # i=1,j=2
        u[1*4 + 3] = u4[3]  # i=1,j=3
        
        # Map v4 to V indices: j*7+k for j in 2,3 and k in col_offset_b, col_offset_b+1
        v[2*7 + col_offset_b] = v4[0]   # j=2, k=col_offset_b
        v[2*7 + col_offset_b+1] = v4[1] # j=2, k=col_offset_b+1
        v[3*7 + col_offset_b] = v4[2]   # j=3, k=col_offset_b
        v[3*7 + col_offset_b+1] = v4[3] # j=3, k=col_offset_b+1
        
        # Map w4 to W indices: k*2+i for k in col_offset_b, col_offset_b+1 and i in 0,1
        w[col_offset_b*2 + 0] = w4[0]
        w[col_offset_b*2 + 1] = w4[1]
        w[(col_offset_b+1)*2 + 0] = w4[2]
        w[(col_offset_b+1)*2 + 1] = w4[3]
        
        all_u.append(u)
        all_v.append(v)
        all_w.append(w)
    
    def add_term_left_2x2x1(u4, v2, w2, col_offset_b):
        """
        Add a schoolbook term for the 2x2x1 part of the left half.
        u4: length 4 for A_left
        v2: length 2 for B column col_offset_b (2x1)
        w2: length 2 for C column col_offset_b (2x1)
        """
        u = [0] * 8
        v = [0] * 28
        w = [0] * 14
        
        u[0] = u4[0]
        u[1] = u4[1]
        u[4] = u4[2]
        u[5] = u4[3]
        
        v[0*7 + col_offset_b] = v2[0]
        v[1*7 + col_offset_b] = v2[1]
        
        w[col_offset_b*2 + 0] = w2[0]
        w[col_offset_b*2 + 1] = w2[1]
        
        all_u.append(u)
        all_v.append(v)
        all_w.append(w)
    
    def add_term_right_2x2x1(u4, v2, w2, col_offset_b):
        """
        Add a schoolbook term for the 2x2x1 part of the right half.
        """
        u = [0] * 8
        v = [0] * 28
        w = [0] * 14
        
        u[0*4 + 2] = u4[0]
        u[0*4 + 3] = u4[1]
        u[1*4 + 2] = u4[2]
        u[1*4 + 3] = u4[3]
        
        v[2*7 + col_offset_b] = v2[0]
        v[3*7 + col_offset_b] = v2[1]
        
        w[col_offset_b*2 + 0] = w2[0]
        w[col_offset_b*2 + 1] = w2[1]
        
        all_u.append(u)
        all_v.append(v)
        all_w.append(w)
    
    # Add terms for left half (A_left * B_top)
    for col_block in range(3):
        col_offset = col_block * 2
        for (u4, v4, w4) in strassen_terms:
            add_term_left(u4, v4, w4, col_offset)
    for (u4, v2, w2) in schoolbook_terms_2x2x1:
        add_term_left_2x2x1(u4, v2, w2, 6)
    
    # Add terms for right half (A_right * B_bottom)
    for col_block in range(3):
        col_offset = col_block * 2
        for (u4, v4, w4) in strassen_terms:
            add_term_right(u4, v4, w4, col_offset)
    for (u4, v2, w2) in schoolbook_terms_2x2x1:
        add_term_right_2x2x1(u4, v2, w2, 6)
    
    rank = len(all_u)
    
    # Build U, V, W as lists of lists
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(28)]
    W = [[0] * rank for _ in range(14)]
    
    for r in range(rank):
        for i in range(8):
            U[i][r] = all_u[r][i]
        for j in range(28):
            V[j][r] = all_v[r][j]
        for k in range(14):
            W[k][r] = all_w[r][k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
