    # Block 3
    u3, v3, w3 = get_schoolbook_4x1()
    rank_offset = 2 * rank_4x4
    for i in range(16):
        for k in range(rank_4x1):
            U_full[i][rank_offset + k] = u3[i][k]
            
        r = i // 4
        c = i % 4
        # B col 8. Global B index = r*9 + 8
        v_idx = r * 9 + 8
        # v3 has 4 rows. Row r corresponds to B row r.
        # But wait, v3 is 4x16.
        # In schoolbook 4x4x1, V[j][col] = 1 if j==i%4?
        # My get_schoolbook_4x1:
        # for i in range(4): # C row
        #   for j in range(4): # A col / B row
        #     U[i*4+j][col] = 1
        #     V[j][col] = 1
        #     W[i][col] = 1
        # So V[j] is for B row j.
        # So for global B row r, we use v3[r].
        
        for k in range(rank_4x1):
            V_full[v_idx][rank_offset + k] = v3[r][k]
            
        # C row 8. Global C index = r*4 + 8
        w_idx = r * 4 + 8
        for k in range(rank_4x1):
            W_full[w_idx][rank_offset + k] = w3[r][k]
