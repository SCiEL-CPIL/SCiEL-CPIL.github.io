"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen coefficients alpha[S][I][J]
    # S is 1..7, I,J are 0..1
    alpha = [[[0]*2 for _ in range(2)] for _ in range(8)]
    # S=1: A00 + A11
    alpha[1][0][0] = 1
    alpha[1][1][1] = 1
    # S=2: A10 + A11
    alpha[2][1][0] = 1
    alpha[2][1][1] = 1
    # S=3: A00
    alpha[3][0][0] = 1
    # S=4: A11
    alpha[4][1][1] = 1
    # S=5: A00 + A01
    alpha[5][0][0] = 1
    alpha[5][0][1] = 1
    # S=6: A10 - A00
    alpha[6][1][0] = 1
    alpha[6][0][0] = -1
    # S=7: A01 - A11
    alpha[7][0][1] = 1
    alpha[7][1][1] = -1
    
    # Outer Strassen coefficients beta[S][J][K]
    beta = [[[0]*2 for _ in range(2)] for _ in range(8)]
    # S=1: B00 + B11
    beta[1][0][0] = 1
    beta[1][1][1] = 1
    # S=2: B00
    beta[2][0][0] = 1
    # S=3: B10 - B11
    beta[3][1][0] = 1
    beta[3][1][1] = -1
    # S=4: B01 - B00
    beta[4][0][1] = 1
    beta[4][0][0] = -1
    # S=5: B11
    beta[5][1][1] = 1
    # S=6: B00 + B01
    beta[6][0][0] = 1
    beta[6][0][1] = 1
    # S=7: B10 + B11
    beta[7][1][0] = 1
    beta[7][1][1] = 1
    
    # Outer Reconstruction gamma[S][I][K]
    gamma = [[[0]*2 for _ in range(2)] for _ in range(8)]
    # C00 = M1 + M4 - M5 + M7
    gamma[1][0][0] = 1
    gamma[4][0][0] = 1
    gamma[5][0][0] = -1
    gamma[7][0][0] = 1
    # C01 = M3 + M5
    gamma[3][0][1] = 1
    gamma[5][0][1] = 1
    # C10 = M2 + M4
    gamma[2][1][0] = 1
    gamma[4][1][0] = 1
    # C11 = M1 - M2 + M3 + M6
    gamma[1][1][1] = 1
    gamma[2][1][1] = -1
    gamma[3][1][1] = 1
    gamma[6][1][1] = 1
    
    # Inner Strassen factors u[t][a][b]
    # t is 1..7, a,b are 0..1
    u = [[[0]*2 for _ in range(2)] for _ in range(8)]
    # P1: (x00 + x11)
    u[1][0][0] = 1
    u[1][1][1] = 1
    # P2: (x10 + x11)
    u[2][1][0] = 1
    u[2][1][1] = 1
    # P3: x00
    u[3][0][0] = 1
    # P4: x11
    u[4][1][1] = 1
    # P5: (x00 + x01)
    u[5][0][0] = 1
    u[5][0][1] = 1
    # P6: (x10 - x00)
    u[6][1][0] = 1
    u[6][0][0] = -1
    # P7: (x01 - x11)
    u[7][0][1] = 1
    u[7][1][1] = -1
    
    # Inner Strassen factors v[t][c][d]
    v = [[[0]*2 for _ in range(2)] for _ in range(8)]
    # P1: (y00 + y11)
    v[1][0][0] = 1
    v[1][1][1] = 1
    # P2: y00
    v[2][0][0] = 1
    # P3: (y10 - y11)
    v[3][1][0] = 1
    v[3][1][1] = -1
    # P4: (y01 - y00)
    v[4][0][1] = 1
    v[4][0][0] = -1
    # P5: y11
    v[5][1][1] = 1
    # P6: (y00 + y01)
    v[6][0][0] = 1
    v[6][0][1] = 1
    # P7: (y10 + y11)
    v[7][1][0] = 1
    v[7][1][1] = 1
    
    # Inner Reconstruction w[t][e][f]
    w = [[[0]*2 for _ in range(2)] for _ in range(8)]
    # Z00 = P1 + P4 - P5 + P7
    w[1][0][0] = 1
    w[4][0][0] = 1
    w[5][0][0] = -1
    w[7][0][0] = 1
    # Z01 = P3 + P5
    w[3][0][1] = 1
    w[5][0][1] = 1
    # Z10 = P2 + P4
    w[2][1][0] = 1
    w[4][1][0] = 1
    # Z11 = P1 - P2 + P3 + P6
    w[1][1][1] = 1
    w[2][1][1] = -1
    w[3][1][1] = 1
    w[6][1][1] = 1
    
    for r in range(rank):
        S = r // 7 + 1
        t = r % 7 + 1
        
        # Fill U
        for i in range(n):
            I = i // 2
            a = i % 2
            for j in range(m):
                J = j // 2
                b = j % 2
                U[i * m + j][r] = alpha[S][I][J] * u[t][a][b]
                
        # Fill V
        for j in range(m):
            J = j // 2
            c = j % 2
            for k in range(p):
                K = k // 2
                d = k % 2
                V[j * p + k][r] = beta[S][J][K] * v[t][c][d]
                
        # Fill W
        for i in range(n):
            I = i // 2
            e = i % 2
            for k in range(p):
                K = k // 2
                f = k % 2
                W[k * n + i][r] = gamma[S][I][K] * w[t][e][f]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
