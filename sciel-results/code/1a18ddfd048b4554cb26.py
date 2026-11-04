for k2 in range(2):
    for i2 in range(2):
        idx2_w = k2 * 2 + i2
        w_val1 = W2[idx2_w][r1]
        ...
        for k2_p in range(2):
            for i2_p in range(2):
                idx2_w_p = k2_p * 2 + i2_p
                w_val2 = W2[idx2_w_p][r2]
                ...
                k4 = 2 * k2 + k2_p
                i4 = 2 * i2 + i2_p
                idx4_w = k4 * 4 + i4
