def construct():
    """
    Rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Strassen's algorithm: 4x4*4x4 has rank 49, and 4x4*4x8 = 2*(4x4*4x4).
    
    U shape: (4*4, rank) = (16, rank)  -- indexed by i*4+j
    V shape: (4*8, rank) = (32, rank)  -- indexed by j*8+k
    W shape: (8*4, rank) = (32, rank)  -- indexed by k*4+i
    """
    n, m, p = 4, 4, 8
    
    # Strassen 2x2 base case: rank 7
    # U: (4, 7) indexed i*2+j, V: (4, 7) indexed j*2+k, W: (4, 7) indexed k*2+i
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Strassen formulas for 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b01-b11)
    # M4 = (a11)(b10-b00)
    # M5 = (a00+a10)(b11)
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    #
    # C00 = M1 - M4 + M5 + M6
    # C01 = M3 + M1 - M5 + M7
    # C10 = M2 + M4 - M1 + M3
    # C11 = M1 - M2 + M6 + M7
    
    def make_strassen_2x2():
        """Returns U, V, W matrices for 2x2*2x2 Strassen, each of shape (4, 7)."""
        rank = 7
        U = [[0]*rank for _ in range(4)]  # i*2+j
        V = [[0]*rank for _ in range(4)]  # j*2+k
        W = [[0]*rank for _ in range(4)]  # k*2+i
        
        # Each term: list of (U_indices_with_coeffs, V_indices_with_coeffs, W_indices_with_coeffs)
        # U index: i*2+j, V index: j*2+k, W index: k*2+i
        
        # M1 = (a00+a11)(b00+b11)
        # U: a00=(0,0)->0, a11=(1,1)->3 => U[0]=1, U[3]=1
        # V: b00=(0,0)->0, b11=(1,1)->3 => V[0]=1, V[3]=1
        # W: contributes to C00(k=0,i=0)->0, C01(k=1,i=0)->1, C10(k=0,i=1)->2, C11(k=1,i=1)->3
        # C00=+M1, C01=+M1, C10=-M1, C11=+M1 => W[0]=1, W[1]=1, W[2]=-1, W[3]=1
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 1:1, 2:-1, 3:1}),   # M1
            ({2:1, 3:1}, {0:1},   {0:0, 1:0, 2:1, 3:-1}),      # M2: (a10+a11)*b00
            ({0:1}, {1:1, 3:-1},  {0:0, 1:1, 2:1, 3:0}),       # M3: a00*(b01-b11)
            ({3:1}, {0:-1, 2:1},  {0:-1, 1:0, 2:1, 3:0}),      # M4: a11*(b10-b00)
            ({0:1, 2:1}, {3:1},   {0:1, 1:-1, 2:0, 3:0}),      # M5: (a00+a10)*b11
            ({2:1, 0:-1}, {0:1, 2:1}, {0:1, 1:0, 2:0, 3:1}),   # M6: (a10-a00)*(b00+b10)
            ({1:1, 3:-1}, {1:1, 3:1}, {0:0, 1:1, 2:0, 3:1}),   # M7: (a01-a11)*(b01+b11)
        ]
        
        for t, (u_c, v_c, w_c) in enumerate(terms):
            for idx, val in u_c.items():
                U[idx][t] = val
            for idx, val in v_c.items():
                V[idx][t] = val
            for idx, val in w_c.items():
                W[idx][t] = val
        return U, V, W
    
    # Build 4x4*4x4 Strassen decomposition (rank 49)
    # Split into 2x2 blocks. Use Strassen at the block level.
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # C blocks: C00, C01, C10, C11 (each 2x2)
    #
    # Block-level Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 - M4 + M5 + M6
    # C01 = M3 + M1 - M5 + M7
    # C10 = M2 + M4 - M1 + M3
    # C11 = M1 - M2 + M6 + M7
    #
    # Each M is a 2x2*2x2 product, which itself uses rank 7.
    # Total: 7 * 7 = 49.
    
    def make_strassen_4x4():
        """Returns U, V, W for 4x4*4x4 Strassen, each of shape (16, 49).
        U indexed by i*4+j, V by j*4+k, W by k*4+i.
        """
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        u2, v2, w2 = make_strassen_2x2()
        
        # Block offsets for 2x2 subblocks within 4x4:
        # Block (br, bc): rows br*2 to br*2+1, cols bc*2 to bc*2+1
        # For U[i*4+j]: block (br,bc) => i in [br*2, br*2+2), j in [bc*2, bc*2+2)
        #   local indices: li = i - br*2, lj = j - bc*2 => local U index = li*2+lj
        # For V[j*4+k]: block (br,bc) => j in [br*2, br*2+2), k in [bc*2, bc*2+2)
        #   local: lj = j - br*2, lk = k - bc*2 => local V index = lj*2+lk
        # For W[k*4+i]: block (br,bc) => k in [br*2, br*2+2), i in [bc*2, bc*2+2)
        #   local: lk = k - br*2, li = i - bc*2 => local W index = lk*2+li
        
        # Strassen terms at block level:
        # Each term has: list of (A_block, sign), list of (B_block, sign), list of (C_block, sign)
        # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        strassen_terms = [
            # (a_blocks_with_signs, b_blocks_with_signs, c_blocks_with_signs)
            # M1 = (A00+A11)*(B00+B11)
            # C: C00+=M1, C01+=M1, C10-=M1, C11+=M1
            ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (1, 1), (2, -1), (3, 1) ]),
            # M2 = (A10+A11)*B00
            # C: C10+=M2, C11-=M2
            ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
            # M3 = A00*(B01-B11)
            # C: C01+=M3, C10+=M3
            ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (2, 1) ]),
            # M4 = A11*(B10-B00)
            # C: C00-=M4, C10+=M4
            ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, -1), (2, 1) ]),
            # M5 = (A00+A10)*B11
            # C: C00+=M5, C01-=M5
            ([ (0, 1), (2, 1) ], [ (3, 1) ], [ (0, 1), (1, -1) ]),
            # M6 = (A10-A00)*(B00+B10)
            # C: C00+=M6, C11+=M6
            ([ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ], [ (0, 1), (3, 1) ]),
            # M7 = (A01-A11)*(B01+B11)
            # C: C01+=M7, C11+=M7
            ([ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ], [ (1, 1), (3, 1) ]),
        ]
        
        for term_idx, (a_blocks, b_blocks, c_blocks) in enumerate(strassen_terms):
            # For this Strassen term, we need to expand each 2x2*2x2 product using the 2x2 Strassen
            # The combined term index in the 49-dim space: term_idx * 7 + sub_term_idx
            # But we need to handle the signs from the block-level Strassen AND the 2x2 Strassen.
            
            # For each combination of a_block and b_block in this term, we form a 2x2*2x2 product
            # with signs, and the result contributes to C blocks with signs.
            
            # Actually, let's think differently. Each Strassen term M_t is a product of sums of A-blocks
            # and sums of B-blocks. The product expands to a sum of individual 2x2*2x2 products.
            # Each individual 2x2*2x2 product uses 7 rank-1 terms (from make_strassen_2x2).
            
            # For term M_t with a_blocks = [(ab, sa), ...] and b_blocks = [(bb, sb), ...]:
            # M_t = sum_{(ab,sa) in a_blocks} sum_{(bb,sb) in b_blocks} sa*sb * (A_ab * B_bb)
            # Each A_ab * B_bb is a 2x2*2x2 product.
            
            # The result M_t contributes to C blocks: sum_{(cb,sc) in c_blocks} sc * C_cb
            
            # So for each (ab, sa) and (bb, sb):
            #   The 2x2*2x2 product A_ab*B_bb produces a 2x2 result.
            #   This result is added to each C_cb with coefficient sa*sb*sc.
            
            # Using the 2x2 Strassen decomposition:
            # sum_{r=0}^{6} u2[local_u, r] * v2[local_v, r] * w2[local_w, r] = (A_ab * B_bb)[local_c]
            # where local_u = li*2+lj for A block ab, local_v = lj*2+lk for B block bb,
            # local_w = lk*2+li for C block cb.
            
            for (ab, sa) in a_blocks:
                ab_r, ab_c = divmod(ab, 2)  # block row, block col
                for (bb, sb) in b_blocks:
                    bb_r, bb_c = divmod(bb, 2)
                    
                    # A block ab: rows ab_r*2..ab_r*2+1, cols ab_c*2..ab_c*2+1
                    # B block bb: rows bb_r*2..bb_r*2+1, cols bb_c*2..bb_c*2+1
                    # For the product A_ab * B_bb to be valid, we need ab_c == bb_r (shared dimension)
                    # In our block structure, this should hold for the Strassen formulas.
                    
                    for (cb, sc) in c_blocks:
                        cb_r, cb_c = divmod(cb, 2)
                        # C block cb: rows cb_r*2..cb_r*2+1, cols cb_c*2..cb_c*2+1
                        # For validity: cb_r == ab_r and cb_c == bb_c
                        
                        coeff = sa * sb * sc
                        if coeff == 0:
                            continue
                        
                        # For each of the 7 rank-1 terms in the 2x2 Strassen:
                        for r in range(7):
                            global_r = term_idx * 7 + r
                            
                            # U: A block ab, local index li*2+lj
                            for li in range(2):
                                for lj in range(2):
                                    local_u = li * 2 + lj
                                    val_u = u2[local_u][r]
                                    if val_u != 0:
                                        global_i = ab_r * 2 + li
                                        global_j = ab_c * 2 + lj
                                        U[global_i * 4 + global_j][global_r] += coeff * val_u
                            
                            # V: B block bb, local index lj*2+lk
                            for lj in range(2):
                                for lk in range(2):
                                    local_v = lj * 2 + lk
                                    val_v = v2[local_v][r]
                                    if val_v != 0:
                                        global_j = bb_r * 2 + lj
                                        global_k = bb_c * 2 + lk
                                        V[global_j * 4 + global_k][global_r] += coeff * val_v
                            
                            # W: C block cb, local index lk*2+li
                            for lk in range(2):
                                for li in range(2):
                                    local_w = lk * 2 + li
                                    val_w = w2[local_w][r]
                                    if val_w != 0:
                                        global_k = cb_r * 2 + lk
                                        global_i = cb_c * 2 + li
                                        W[global_k * 4 + global_i][global_r] += coeff * val_w
        
        return U, V, W
    
    # Now build the 4x4 * 4x8 decomposition
    # Split B into two 4x4 blocks: B_left (cols 0-3) and B_right (cols 4-7)
    # C = A * B = [A * B_left | A * B_right]
    # Each A * B_left and A * B_right is a 4x4*4x4 product, rank 49 each.
    # Total rank = 98.
    
    u4, v4, w4 = make_strassen_4x4()
    
    rank = 98
    U = [[0]*rank for _ in range(16)]   # i*4+j
    V = [[0]*rank for _ in range(32)]   # j*8+k
    W = [[0]*rank for _ in range(32)]   # k*4+i
    
    # First half: B_left (k=0..3), uses rank terms 0..48
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i*4+j][r] = u4[i*4+j][r]
    
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j*8+k][r] = v4[j*4+k][r]
    
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k*4+i][r] = w4[k*4+i][r]
    
    # Second half: B_right (k=4..7), uses rank terms 49..97
    for i in range(4):
        for j in range(4):
            for r in range(49):
                U[i*4+j][49+r] = u4[i*4+j][r]
    
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j*8+(4+k)][49+r] = v4[j*4+k][r]
    
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[(4+k)*4+i][49+r] = w4[k*4+i][r]
    
    return {'U': U, 'V': V, 'W': W}
