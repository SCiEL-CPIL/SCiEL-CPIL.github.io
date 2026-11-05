    U2, V2, W2 = build_2x2x2_strassen()
    base = 4
    v2_map = {0: 1, 1: 2, 2: 4, 3: 5}
    w2_map = {0: 2, 1: 4, 2: 3, 3: 5}
    for c in range(7):
        for r in range(4):
            if U2[r][c] != 0:
                U[r][base + c] += U2[r][c]
        for r in range(4):
            if V2[r][c] != 0:
                V[v2_map[r]][base + c] += V2[r][c]
        for r in range(4):
            if W2[r][c] != 0:
                W[w2_map[r]][base + c] += W2[r][c]
