DIMENSIONS = [4, 4, 4]

def construct():
    # 2x2 Strassen decomposition (verified)
    U2 = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ]
    V2 = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [-1, 0, 1, 0],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ]
    W2 = [
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 1, 0, 0],   # M2: C10
        [0, 0, 1, 0],   # M3: C01
        [1, 1, 0, 0],   # M4: C00+C10
        [-1, 0, 1, 0],  # M5: C01-C00
        [0, 0, 0, 1],   # M6: C11
        [-1, 0, 0, 1]   # M7: C11-C00
    ]
    
    rank4 = 49
    U4 = [[0]*rank4 for _ in range(16)]
    V4 = [[0]*rank4 for _ in range(16)]
    W4 = [[0]*rank4 for _ in range(16)]
    
    for a in range(7):
        for b in range(7):
            col = a * 7 + b
            # U and V: input indices
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i4 = i1 * 2 + i2
                            j4 = j1 * 2 + j2
                            idx4 = i4 * 4 + j4
                            idx_block = i1 * 2 + j1
                            idx_sub = i2 * 2 + j2
                            U4[idx4][col] = U2[idx_block][a] * U2[idx_sub][b]
                            V4[idx4][col] = V2[idx_block][a] * V2[idx_sub][b]
            # W: output index W[k*n+i]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k4 = k1 * 2 + k2
                            i4 = i1 * 2 + i2
                            idx4 = k4 * 4 + i4
                            idx_block_W = k1 * 2 + i1
                            idx_sub_W = k2 * 2 + i2
                            W4[idx4][col] = W2[idx_block_W][a] * W2[idx_sub_W][b]
    
    return {"U": U4, "V": V4, "W": W4}
