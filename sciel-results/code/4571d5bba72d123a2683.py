def construct():
    # Strassen 2x2x2 (rank 7)
    U1 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V1 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    W1 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    # 2x2x3 (rank 11) = schoolbook(2x2x1) + Strassen(2x2x2)
    U2_part1 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    V2_part1 = [
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    W2_part1 = [
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    
    U2_part2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    V2_part2 = [
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    W2_part2 = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    U2 = [a + b for a, b in zip(U2_part1, U2_part2)]
    V2 = [a + b for a, b in zip(V2_part1, V2_part2)]
    W2 = [a + b for a, b in zip(W2_part1, W2_part2)]
    
    rank = 77
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]
    W = [[0] * rank for _ in range(24)]
    
    for i1 in range(2):
        for j1 in range(2):
            r1 = i1 * 2 + j1
            for i2 in range(2):
                for j2 in range(2):
                    r2 = i2 * 2 + j2
                    row = r1 * 4 + r2
                    for c1 in range(7):
                        for c2 in range(11):
                            col = c1 * 11 + c2
                            U[row][col] = U1[r1][c1] * U2[r2][c2]
    
    for j1 in range(2):
        for k1 in range(2):
            r1 = j1 * 2 + k1
            for j2 in range(2):
                for k2 in range(3):
                    r2 = j2 * 3 + k2
                    row = r1 * 6 + r2
                    for c1 in range(7):
                        for c2 in range(11):
                            col = c1 * 11 + c2
                            V[row][col] = V1[r1][c1] * V2[r2][c2]
    
    for k1 in range(2):
        for i1 in range(2):
            r1 = k1 * 2 + i1
            for k2 in range(3):
                for i2 in range(2):
                    r2 = k2 * 2 + i2
                    row = r1 * 4 + r2
                    for c1 in range(7):
                        for c2 in range(11):
                            col = c1 * 11 + c2
                            W[row][col] = W1[r1][c1] * W2[r2][c2]
    
    return {"U": U, "V": V, "W": W}
