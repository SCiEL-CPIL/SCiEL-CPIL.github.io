    # Copy 4x4x4 part (first 49 columns)
    U[:, :49] = U4
    # V4 has rows indexed by j*4+k. V has rows indexed by j*5+k.
    # We need to map V4 rows to correct V rows.
    for j in range(4):
        for k in range(4):
            V[j*5 + k, :49] = V4[j*4 + k, :49]
    W[:16, :49] = W4  # W4 is 16x49, corresponds to C[i,0..3]
