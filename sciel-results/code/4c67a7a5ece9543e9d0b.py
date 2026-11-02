"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: 20 terms for j=0 (A[:,0] * B[0,:])
    for i in range(n):
        for k in range(p):
            U[i * m][r] = 1
            V[k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    
    # Part 2: 35 terms for j=1,2 using Strassen on 2x2x2 blocks
    for t in range(5):
        for s in range(7):
            if s == 0:
                U[1][r] = 1
                U[5][r] = 1
                V[10 + 2*t][r] = 1
                V[21 + 2*t][r] = 1
                W[4*t][r] = 1
                W[4*t + 3][r] = 1
            elif s == 1:
                U[4][r] = 1
                U[5][r] = 1
                V[10 + 2*t][r] = 1
                W[4*t + 2][r] = 1
            elif s == 2:
                U[1][r] = 1
                V[11 + 2*t][r] = 1
                V[21 + 2*t][r] = -1
                W[4*t + 1][r] = 1
            elif s == 3:
                U[5][r] = 1
                V[20 + 2*t][r] = 1
                V[10 + 2*t][r] = -1
                W[4*t][r] = 1
                W[4*t + 2][r] = 1
            elif s == 4:
                U[1][r] = 1
                U[2][r] = 1
                V[21 + 2*t][r] = 1
                W[4*t][r] = -1
                W[4*t + 1][r] = 1
            elif s == 5:
                U[4][r] = 1
                U[1][r] = -1
                V[10 + 2*t][r] = 1
                V[11 + 2*t][r] = 1
                W[4*t + 3][r] = 1
            elif s == 6:
                U[2][r] = 1
                U[5][r] = -1
                V[20 + 2*t][r] = 1
                V[21 + 2*t][r] = 1
                W[4*t][r] = 1
                W[4*t + 3][r] = 1
            r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
