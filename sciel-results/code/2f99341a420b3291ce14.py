def construct():
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 decomposition
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0]
    ]
    
    # Build 4x4x4 decomposition via tensor product
    # U4[i*4+j, c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    rank44 = 49
    U4 = [[0]*rank44 for _ in range(16)]
    V4 = [[0]*rank44 for _ in range(16)]
    W4 = [[0]*rank44 for _ in range(16)]
    
    for i in range(4):
        i1, i2 = i // 2, i % 2
        for j in range(4):
            j1, j2 = j // 2, j % 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    U4[i*4+j][c] = U2[i1*2+j1][c1] * U2[i2*2+j2][c2]
    
    for j in range(4):
        j1, j2 = j // 2, j % 2
        for k in range(4):
            k1, k2 = k // 2, k % 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    V4[j*4+k][c] = V2[j1*2+k1][c1] * V2[j2*2+k2][c2]
    
    for k in range(4):
        k1, k2 = k // 2, k % 2
        for i in range(4):
            i1, i2 = i // 2, i % 2
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    W4[k*4+i][c] = W2[k1*2+i1][c1] * W2[k2*2+i2][c2]
    
    # Now build 4x4x8 by concatenating two 4x4x4 decompositions
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # First 49 terms: k in 0..3
    for row in range(16):
        for c in range(49):
            U[row][c] = U4[row][c]
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*8+k][c] = V4[j*4+k][c]
    
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*4+i][c] = W4[k*4+i][c]
    
    # Second 49 terms: k in 4..7
    for row in range(16):
        for c in range(49):
            U[row][49+c] = U4[row][c]
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*8+(k+4)][49+c] = V4[j*4+k][c]
    
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[(k+4)*4+i][49+c] = W4[k*4+i][c]
    
    return {"U": U, "V": V, "W": W}
