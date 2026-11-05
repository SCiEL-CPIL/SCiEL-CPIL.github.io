def construct():
    n, m, p = 4, 5, 5
    rank = 66  # 49 + 16 + 1
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4 x 4x4 (Strassen, rank 49)
    # Strassen 2x2x2 terms
    strassen_2x2 = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: (i,j) -> coeff, v: (j,k) -> coeff, w: (i,k) -> coeff
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # For 4x4x4, each 2x2x2 term expands to 7 scalar terms
    # Block indices: (bi, bj, bk) where bi,bj,bk in {0,1} for 2x2 blocks
    # Within each block, local indices (li, lj, lk) in {0,1}
    
    for (u_block, v_block, w_block) in strassen_2x2:
        for (u_local, v_local, w_local) in strassen_2x2:
            # This is a scalar term in the 4x4x4 decomposition
            # u: global (bi*2+li, bj*2+lj) for each (bi,bj) in u_block and (li,lj) in u_local
            # v: global (bj*2+lj, bk*2+lk) for each (bj,bk) in v_block and (lj,lk) in v_local
            # w: global (bi*2+li, bk*2+lk) for each (bi,bk) in w_block and (li,lk) in w_local
            
            # Wait, this isn't quite right. Let me think again.
            # In the 4x4x4 Strassen, we have 7 "block" terms, each of which is a 2x2x2 multiplication.
            # For each block term, we have 7 scalar terms.
            # The block term (u_block, v_block, w_block) defines which 2x2 blocks are involved.
            # The scalar term (u_local, v_local, w_local) defines which entries within those blocks.
            
            # u_block maps block indices (BI, BJ) to coefficients
            # u_local maps local indices (LI, LJ) to coefficients
            # Global u index: (BI*2+LI, BJ*2+LJ)
            
            # Similarly for v and w.
            
            # But we need to be careful: the block indices for u, v, w must be consistent.
            # u_block has entries (BI, BJ), v_block has entries (BJ, BK), w_block has entries (BI, BK).
            # For a valid term, the BJ in u_block must match the BJ in v_block, etc.
            
            # Actually, in Strassen, the block structure is:
            # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
            # u_block entries are (BI, BJ) where BI, BJ in {0,1}
            # v_block entries are (BJ, BK) where BJ, BK in {0,1}
            # w_block entries are (BI, BK) where BI, BK in {0,1}
            
            # For the term to be valid, we need consistency in the block indices.
            # But in the Strassen decomposition, each term is self-consistent.
            
            # Let me just enumerate all combinations and see which ones are valid.
            
            # Actually, I think the correct way is:
            # For each pair of Strassen terms (outer, inner), the global term is:
            # u[BI*2+LI, BJ*2+LJ] = u_block[BI,BJ] * u_local[LI,LJ]
            # v[BJ*2+LJ, BK*2+LK] = v_block[BJ,BK] * v_local[LJ,LK]
            # w[BI*2+LI, BK*2+LK] = w_block[BI,BK] * w_local[LI,LK]
            
            # But this only makes sense if the block indices are consistent.
            # In the Strassen decomposition, for a given outer term, the block indices are fixed.
            # For example, outer term 0: u_block = {(0,0):1, (1,1):1}, v_block = {(0,0):1, (1,1):1}, w_block = {(0,0):1, (1,1):1}
            # This means BI=BJ for u, BJ=BK for v, BI=BK for w.
            # So BI=BJ=BK.
            
            # For the inner term, LI, LJ, LK are the local indices.
            # The global indices are:
            # u: (BI*2+LI, BJ*2+LJ) = (BI*2+LI, BI*2+LJ) since BJ=BI
            # v: (BJ*2+LJ, BK*2+LK) = (BI*2+LJ, BI*2+LK) since BJ=BK=BI
            # w: (BI*2+LI, BK*2+LK) = (BI*2+LI, BI*2+LK) since BK=BI
            
            # But this is only for outer term 0. Other terms have different block index relationships.
            
            # I think I'm overcomplicating this. Let me just directly construct the 49 terms.
            pass
    
    # This approach is getting too complex. Let me try a different strategy.
    
    # Actually, let me just use schoolbook for the 4x4x4 part (rank 64) and see if the total is 64+16+1=81.
    # That's still better than 100.
    
    # Let me restart with a simpler approach.
    return None
