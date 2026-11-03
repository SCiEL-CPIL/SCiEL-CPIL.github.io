    for r in range(4):
        # B[r, 4] index in 4x5 part is r*5+4
        idx_445 = r * 5 + 4
        for k in range(16):
            V[idx_445][49+k] = V41[r][k]
