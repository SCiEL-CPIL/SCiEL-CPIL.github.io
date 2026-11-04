    # Block 1: columns 0-48 (4x4x4 Strassen)
    U[:, :49] = U444
    for k in range(4):
        for j in range(4):
            V[k*9 + j, :49] = V444[k*4 + j, :49]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, :49] = W444[k*4 + i, :49] # W444 index is j*4+i?
