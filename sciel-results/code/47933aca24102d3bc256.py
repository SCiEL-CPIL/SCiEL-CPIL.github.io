    for k0 in range(2):
        for i0 in range(2):
            for k1 in range(2):
                for i1 in range(2):
                    k = 2 * k0 + k1
                    i = 2 * i0 + i1
                    w_val = W2[k0 * 2 + i0, r0] * W2[k1 * 2 + i1, r1]
                    W[k * 4 + i, col] = w_val
    