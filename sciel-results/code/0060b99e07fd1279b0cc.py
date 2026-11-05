"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2
    # U2, V2, W2 are 4x7
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
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
    
    # Schoolbook decomposition for 3x3
    # U3, V3, W3 are 9x27
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*3+i][col] = 1
                col += 1
                
    # Tensor product
    # Rank = 7 * 27 = 189
    rank = 7 * 27
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    for c1 in range(7):
        for c2 in range(27):
            c = c1 * 27 + c2
            for i1 in range(4):
                u1 = U2[i1][c1]
                if u1 != 0:
                    for i2 in range(9):
                        u3 = U3[i2][c2]
                        if u3 != 0:
                            U[i1*9 + i2][c] = u1 * u3
                            
            for j1 in range(4):
                v1 = V2[j1][c1]
                if v1 != 0:
                    for j2 in range(9):
                        v3 = V3[j2][c2]
                        if v3 != 0:
                            V[j1*9 + j2][c] = v1 * v3
                            
            for k1 in range(4):
                w1 = W2[k1][c1]
                if w1 != 0:
                    for k2 in range(9):
                        w3 = W3[k2][c2]
                        if w3 != 0:
                            W[k1*9 + k2][c] = w1 * w3
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
