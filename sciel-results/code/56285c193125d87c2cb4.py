import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition data
    # U_s[i, j] for X matrix, V_s[j, k] for Y matrix
    # Indices 0..3 correspond to 00, 01, 10, 11
    U_strassen = [
        [1, 0, 0, 1],  # s=1
        [0, 0, 1, 1],  # s=2
        [1, 0, 0, 0],  # s=3
        [0, 0, 0, 1],  # s=4
        [1, 1, 0, 0],  # s=5
        [-1, 0, 1, 0], # s=6
        [0, 1, 0, -1]  # s=7
    ]
    V_strassen = [
        [1, 0, 0, 1],  # s=1
        [1, 0, 0, 0],  # s=2
        [0, 1, 0, -1], # s=3
        [-1, 0, 1, 0], # s=4
        [0, 0, 0, 1],  # s=5
        [1, 1, 0, 0],  # s=6
        [0, 0, 1, 1]   # s=7
    ]
    
    # Gamma[s][i][k] for C_ik = sum_s Gamma[s][i][k] * T_s
    Gamma = [[[0]*2 for _ in range(2)] for _ in range(7)]
    # s=1
    Gamma[0][0][0] = 1
    Gamma[0][1][1] = 1
    # s=2
    Gamma[1][0][1] = 1
    # s=3
    Gamma[2][1][0] = 1
    Gamma[2][1][1] = -1
    # s=4
    Gamma[3][0][0] = -1
    Gamma[3][0][1] = 1
    # s=5
    Gamma[4][1][0] = 1
    Gamma[4][1][1] = 1
    # s=6
    Gamma[5][0][0] = 1
    Gamma[5][1][1] = 1
    # s=7
    Gamma[6][0][0] = 1
    
    # Outer Strassen linear forms for blocks
    # X_r[u][v] for A blocks, Y_r[u][v] for B blocks
    # u, v in {0, 1}
    X_outer = [
        [[1, 0], [0, 1]], # r=1: A00 + A11
        [[0, 0], [1, 1]], # r=2: A10 + A11
        [[1, 0], [0, 0]], # r=3: A00
        [[0, 0], [0, 1]], # r=4: A11
        [[1, 1], [0, 0]], # r=5: A00 + A01
        [[-1, 0], [1, 0]],# r=6: A10 - A00
        [[0, 1], [0, -1]] # r=7: A01 - A11
    ]
    Y_outer = [
        [[1, 0], [0, 1]], # r=1: B00 + B11
        [[1, 0], [0, 0]], # r=2: B00
        [[0, 1], [0, -1]],# r=3: B01 - B11
        [[0, 0], [1, -1]],# r=4: B10 - B00
        [[0, 0], [0, 1]], # r=5: B11
        [[1, 1], [0, 0]], # r=6: B00 + B01
        [[0, 0], [1, 1]]  # r=7: B10 + B11
    ]
    
    # Alpha[r][i1][k1] for C_block[i1, k1] = sum_r Alpha[r][i1][k1] * M_r
    Alpha = [[[0]*2 for _ in range(2)] for _ in range(7)]
    # r=1
    Alpha[0][0][0] = 1
    Alpha[0][1][1] = 1
    # r=2
    Alpha[1][0][1] = 1
    # r=3
    Alpha[2][1][0] = 1
    # r=4
    Alpha[3][0][0] = -1
    Alpha[3][0][1] = 1
    # r=5
    Alpha[4][1][0] = 1
    Alpha[4][1][1] = 1
    # r=6
    Alpha[5][0][0] = 1
    Alpha[5][1][1] = 1
    # r=7
    Alpha[6][0][0] = 1
    Alpha[6][1][0] = 1 # Wait, C10 = M3 + M5. C11 = M1 - M3 + M5 + M6.
    # Let's re-verify Alpha for r=7.
    # C00 = M1 - M4 + M6 + M7 -> Alpha[6][0][0] = 1
    # C01 = M2 + M4 -> No M7
    # C10 = M3 + M5 -> No M7
    # C11 = M1 - M3 + M5 + M6 -> No M7
    # So Alpha[6] should only be non-zero at [0][0].
    Alpha[6][0][0] = 1
    Alpha[6][1][0] = 0 # Corrected
    
    for r in range(7):
        for s in range(7):
            R = r * 7 + s
            for u in range(2):
                for v in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            # U index
                            i = u * 2 + i2
                            j = v * 2 + j2
                            I = i * 4 + j
                            U[I][R] += U_strassen[s][i2 * 2 + j2] * X_outer[r][u][v]
                            
                            # V index
                            # Y_r[j2, k2] = sum Y_outer[r][v][w] B[v*2+j2, w*2+k2]
                            # L_B = sum V_strassen[s][j2, k2] Y_r[j2, k2]
                            # Coeff for B_{v*2+j2, w*2+k2} is V_strassen[s][j2, k2] * Y_outer[r][v][w]
                            # We need to fill V for all j, k
                            # Loop w and k2
                            for w in range(2):
                                for k2 in range(2):
                                    j_idx = v * 2 + j2
                                    k_idx = w * 2 + k2
                                    J = j_idx * 4 + k_idx
                                    V[J][R] += V_strassen[s][j2 * 2 + k2] * Y_outer[r][v][w]
                                    
                            # W index
                            # W depends on i1, k1 (block indices) and i2, k2 (intra indices)
                            # But W is constant for a given R and output K.
                            # K = k*4 + i.
                            # i = i1*2 + i2, k = k1*2 + k2.
                            # We can just fill W for all K.
                            for i1 in range(2):
                                for k1 in range(2):
                                    i = i1 * 2 + i2
                                    k = k1 * 2 + k2
                                    K = k * 4 + i
                                    W[K][R] = Alpha[r][i1][k1] * Gamma[s][i2][k2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
