import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin

    # Strassen 2x2 factors
    U2 = [
        [1, 0, 0, 1, 1, -1, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 1, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3 + j][col] = 1
                V3[j*3 + k][col] = 1
                W3[k*3 + i][col] = 1
                col += 1

    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for r2 in range(7):
        if time.time() > deadline:
            break
        for r3 in range(27):
            c = r2 * 27 + r3
            for row in range(36):
                i_u = row // 6
                j_u = row % 6
                i2_u = i_u // 3
                i3_u = i_u % 3
                j2_u = j_u // 3
                j3_u = j_u % 3
                
                u2 = U2[i2_u * 2 + j2_u][r2]
                u3 = U3[i3_u * 3 + j3_u][r3]
                U[row][c] = u2 * u3
                
                j_v = row // 6
                k_v = row % 6
                j2_v = j_v // 3
                j3_v = j_v % 3
                k2_v = k_v // 3
                k3_v = k_v % 3
                
                v2 = V2[j2_v * 2 + k2_v][r2]
                v3 = V3[j3_v * 3 + k3_v][r3]
                V[row][c] = v2 * v3
                
                k_w = row // 6
                i_w = row % 6
                k2_w = k_w // 3
                k3_w = k_w % 3
                i2_w = i_w // 3
                i3_w = i_w % 3
                
                w2 = W2[k2_w * 2 + i2_w][r2]
                w3 = W3[k3_w * 3 + i3_w][r3]
                W[row][c] = w2 * w3
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
