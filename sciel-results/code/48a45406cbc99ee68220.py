    for idx in range(16):
        i = idx // 4
        j = idx % 4
        c = 49 + idx
        U[i*4+j, c] = 1
        V[j*5+4, c] = 1
        W[4*4+i, c] = 1
