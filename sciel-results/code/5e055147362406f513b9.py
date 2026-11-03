"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 6 * 7 = 42
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Decomposition of M_{1,2,3} (Rank 6)
    # U1[j1, c1], V1[j1*3+k1, c1], W1[k1, c1]
    # c1 = k1 * 2 + j1
    
    # Decomposition of M_{2,2,2} (Rank 7, Strassen)
    # U2[i2*2+j2, c2], V2[j2*2+k2, c2], W2[k2*2+i2, c2]
    
    # Precompute Strassen matrices
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # P1: (X00+X11)(Z00+Z11)
    U2[0][0] += 1; U2[3][0] += 1
    V2[0][0] += 1; V2[3][0] += 1
    W2[0][0] += 1; W2[3][0] += 1
    
    # P2: (X10+X11)Z00
    U2[2][1] += 1; U2[3][1] += 1
    V2[0][1] += 1
    W2[2][1] += 1; W2[3][1] -= 1
    
    # P3: X00(Z01-Z11)
    U2[0][2] += 1
    V2[1][2] += 1; V2[3][2] -= 1
    W2[1][2] += 1; W2[3][2] += 1
    
    # P4: X11(Z10-Z00)
    U2[3][3] += 1
    V2[2][3] += 1; V2[0][3] -= 1
    W2[0][3] += 1; W2[2][3] += 1
    
    # P5: (X00+X01)Z11
    U2[0][4] += 1; U2[1][4] += 1
    V2[3][4] += 1
    W2[0][4] -= 1; W2[1][4] += 1
    
    # P6: (X10-X00)(Z00+Z01)
    U2[2][5] += 1; U2[0][5] -= 1
    V2[0][5] += 1; V2[1][5] += 1
    W2[3][5] += 1
    
    # P7: (X01-X11)(Z10+Z11)
    U2[1][6] += 1; U2[3][6] -= 1
    V2[2][6] += 1; V2[3][6] += 1
    W2[0][6] += 1; W2[3][6] += 1
    
    # Fill U, V, W
    for c1 in range(6):
        k1 = c1 // 2
        j1 = c1 % 2
        
        # U1[j1, c1] is 1
        u1_val = 1
        
        # V1[j1*3+k1, c1] is 1
        v1_row = j1 * 3 + k1
        v1_val = 1
        
        # W1[k1, c1] is 1
        w1_val = 1
        
        for c2 in range(7):
            r = c1 * 7 + c2
            
            # U2[i2*2+j2, c2]
            # U[i*4+j, r] = U1[j1, c1] * U2[i2*2+j2, c2]
            # i = i2, j = j1*2 + j2
            # u = i2*4 + j1*2 + j2
            for i2 in range(2):
                for j2 in range(2):
                    u2_val = U2[i2*2 + j2][c2]
                    if u2_val != 0:
                        u = i2 * 4 + j1 * 2 + j2
                        U[u][r] = u1_val * u2_val
                        
            # V2[j2*2+k2, c2]
            # V[j*6+k, r] = V1[j1*3+k1, c1] * V2[j2*2+k2, c2]
            # j = j1*2 + j2, k = k1*2 + k2
            # v = j*6 + k = (j1*2+j2)*6 + (k1*2+k2)
            for j2 in range(2):
                for k2 in range(2):
                    v2_val = V2[j2*2 + k2][c2]
                    if v2_val != 0:
                        v = (j1 * 2 + j2) * 6 + (k1 * 2 + k2)
                        V[v][r] = v1_val * v2_val
                        
            # W2[k2*2+i2, c2]
            # W[k*2+i, r] = W1[k1, c1] * W2[k2*2+i2, c2]
            # k = k1*2 + k2, i = i2
            # w = k*2 + i = (k1*2+k2)*2 + i2
            for k2 in range(2):
                for i2 in range(2):
                    w2_val = W2[k2*2 + i2][c2]
                    if w2_val != 0:
                        w = (k1 * 2 + k2) * 2 + i2
                        W[w][r] = w1_val * w2_val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
