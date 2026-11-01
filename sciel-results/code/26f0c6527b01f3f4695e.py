"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Rank is 7 * (p/2) = 7 * 7 = 49
    rank = 7 * (p // 2)
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2 multiplication
    # u[i][j][r], v[j][k][r], w[k][i][r]
    # i, j, k in {0, 1}
    # r in {0, ..., 6}
    
    u = [[[0]*7 for _ in range(2)] for _ in range(2)]
    v = [[[0]*7 for _ in range(2)] for _ in range(2)]
    w = [[[0]*7 for _ in range(2)] for _ in range(2)]
    
    # r=0: M1 = (A00+A11)(B00+B11)
    u[0][0][0] = 1; u[1][1][0] = 1
    v[0][0][0] = 1; v[1][1][0] = 1
    w[0][0][0] = 1; w[1][1][0] = 1
    
    # r=1: M2 = (A10+A11)B00
    u[1][0][1] = 1; u[1][1][1] = 1
    v[0][0][1] = 1
    w[1][0][1] = 1; w[1][1][1] = -1
    
    # r=2: M3 = A00(B01-B11)
    u[0][0][2] = 1
    v[0][1][2] = 1; v[1][1][2] = -1
    w[0][1][2] = 1; w[1][1][2] = 1
    
    # r=3: M4 = A11(B10-B00)
    u[1][1][3] = 1
    v[1][0][3] = 1; v[0][0][3] = -1
    w[0][0][3] = 1; w[1][0][3] = 1
    
    # r=4: M5 = (A00+A01)B11
    u[0][0][4] = 1; u[0][1][4] = 1
    v[1][1][4] = 1
    w[0][0][4] = -1; w[0][1][4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    u[1][0][5] = 1; u[0][0][5] = -1
    v[0][0][5] = 1; v[0][1][5] = 1
    w[1][1][5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    u[0][1][6] = 1; u[1][1][6] = -1
    v[1][0][6] = 1; v[1][1][6] = 1
    w[0][0][6] = 1
    
    # Fill U, V, W
    for l in range(p // 2):
        for r in range(7):
            col = l * 7 + r
            # Fill U
            for i in range(2):
                for j in range(2):
                    if u[i][j][r] != 0:
                        U[i * m + j][col] = u[i][j][r]
            # Fill V
            for j in range(2):
                for k in range(2):
                    if v[j][k][r] != 0:
                        global_k = 2 * l + k
                        V[j * p + global_k][col] = v[j][k][r]
            # Fill W
            for k in range(2):
                for i in range(2):
                    if w[k][i][r] != 0:
                        global_k = 2 * l + k
                        W[global_k * n + i][col] = w[k][i][r]
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
