import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 2x2 Strassen decomposition
    # U: (4, 7), V: (4, 7), W: (4, 7)
    U2 = [
        [1, 0, 1, 0, 1, -1, 1],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Kronecker product logic
    # Indices for 4x4:
    # i = i1 * 2 + i2, j = j1 * 2 + j2, k = k1 * 2 + k2
    # r = r1 * 7 + r2
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    u_row = i1 * 8 + i2 * 4 + j1 * 2 + j2
                    # Actually i*m + j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            val = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                            if val != 0:
                                U[u_row][r] = val
                            
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    v_row = j1 * 8 + j2 * 4 + k1 * 2 + k2
                    # j*p + k = (j1*2+j2)*4 + (k1*2+k2) = j1*8 + j2*4 + k1*2 + k2
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            val = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                            if val != 0:
                                V[v_row][r] = val
                                
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    w_row = k1 * 8 + k2 * 4 + i1 * 2 + i2
                    # k*n + i = (k1*2+k2)*4 + (i1*2+i2) = k1*8 + k2*4 + i1*2 + i2
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            val = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
                            if val != 0:
                                W[w_row][r] = val
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
