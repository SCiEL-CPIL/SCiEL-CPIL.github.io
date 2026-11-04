def get_4x4_strassen():
    # 2x2 Strassen
    U_2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, -1, 1, -1, 0, 0, 0]
    ])
    
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    for k1 in range(7):
        for k2 in range(7):
            r = k1 * 7 + k2
            
            # U_4
            for la in range(4):
                c1 = U_2[la, k1]
                if c1 == 0: continue
                br, bc = divmod(la, 2)
                for r_loc in range(2):
                    for c_loc in range(2):
                        gi = br * 2 + r_loc
                        gj = bc * 2 + c_loc
                        # Inner term k2 applies to the 2x2 sub-block
                        # The 2x2 sub-block at (gi, gj) in A corresponds to local index (r_loc, c_loc) in the 2x2 block?
                        # No, the 4x4 A is composed of 2x2 blocks.
                        # The "inner" Strassen acts on the 2x2 blocks.
                        # So for a fixed outer term k1, we have a 4x4 matrix A_k1.
                        # A_k1 is a 2x2 matrix of 2x2 blocks.
                        # Let's say A_k1 is [[A_k1_00, A_k1_01], [A_k1_10, A_k1_11]].
                        # Each A_k1_ij is 2x2.
                        # The inner Strassen term k2 picks a linear combo of the *entries* of these 2x2 blocks?
                        # No, the inner Strassen multiplies the 2x2 blocks.
                        
                        # Actually, the coefficient for A[i,j] in U_4[r] is:
                        # sum_{block_l} U_2[block_l, k1] * U_2x2[local_idx, k2]
                        # Where local_idx corresponds to the position of A[i,j] within the 2x2 block.
                        
                        # Let's map global (gi, gj) to block (br, bc) and local (r_loc, c_loc).
                        # br = gi // 2, bc = gj // 2.
                        # r_loc = gi % 2, c_loc = gj % 2.
                        # Local index in 2x2 block: r_loc * 2 + c_loc.
                        
                        # The block index la corresponds to (br, bc).
                        # So we need U_2[la, k1] where la = br * 2 + bc.
                        
                        # And U_2x2[r_loc * 2 + c_loc, k2].
                        
                        # So:
                        # coeff = U_2[br*2+bc, k1] * U_2[r_loc*2+c_loc, k2]
                        
                        U_4[gi * 4 + gj, r] += c1 * U_2[r_loc * 2 + c_loc, k2]
                        
            # V_4
            for lb in range(4):
                c1 = V_2[lb, k1]
                if c1 == 0: continue
                br, bc = divmod(lb, 2)
                for r_loc in range(2):
                    for c_loc in range(2):
                        gj = br * 2 + r_loc
                        gk = bc * 2 + c_loc
                        V_4[gj * 4 + gk, r] += c1 * V_2[r_loc * 2 + c_loc, k2]
                        
            # W_4
            # W is indexed by (k, i).
            # Global (gk, gi).
            # Block (bk, bi). Local (r_loc, c_loc) where r_loc is k-local, c_loc is i-local?
            # W_2 is indexed by (j, i) or (k, i)?
            # In 2x2, W_2[row, col]. Row is (j, i) index?
            # Let's assume W_2[row, k] where row = j*2+i.
            # So for global (gk, gi):
            # bk = gk // 2, bi = gi // 2.
            # r_loc = gk % 2, c_loc = gi % 2.
            # Local index = r_loc * 2 + c_loc.
            # Block index lw = bk * 2 + bi.
            
            for lw in range(4):
                c1 = W_2[lw, k1]
                if c1 == 0: continue
                bk, bi = divmod(lw, 2)
                for r_loc in range(2):
                    for c_loc in range(2):
                        gk = bk * 2 + r_loc
                        gi = bi * 2 + c_loc
                        W_4[gk * 4 + gi, r] += c1 * W_2[r_loc * 2 + c_loc, k2]
                        
    return U_4, V_4, W_4
