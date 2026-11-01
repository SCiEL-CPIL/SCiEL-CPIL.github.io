"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    # Rank-39 decomposition using 5 Strassen blocks (2x2x2) and 1 schoolbook block (2x2x1)
    n, m, p = DIMENSIONS
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def strassen_block():
        U_b = [[0]*7 for _ in range(4)]
        V_b = [[0]*7 for _ in range(4)]
        W_b = [[0]*7 for _ in range(4)]
        
        U_b[0][0] = 1; U_b[3][0] = 1
        V_b[2][0] = 1; V_b[3][0] = 1
        W_b[0][0] = 1; W_b[1][0] = 1; W_b[2][0] = 1; W_b[3][0] = 1
        
        U_b[2][1] = 1; U_b[3][1] = 1
        V_b[0][1] = 1; V_b[1][1] = 1
        W_b[0][1] = 1; W_b[1][1] = 1; W_b[2][1] = 1; W_b[3][1] = 1
        
        U_b[0][2] = 1; U_b[1][2] = 1
        V_b[2][2] = 1; V_b[3][2] = 1
        W_b[0][2] = 1; W_b[1][2] = 1; W_b[2][2] = 1; W_b[3][2] = 1
        
        U_b[0][3] = 1; U_b[2][3] = -1
        V_b[0][3] = 1; V_b[2][3] = 1
        W_b[0][3] = 1; W_b[1][3] = 1; W_b[2][3] = 1; W_b[3][3] = 1
        
        U_b[1][4] = 1; U_b[3][4] = -1
        V_b[1][4] = 1; V_b[3][4] = 1
        W_b[0][4] = 1; W_b[1][4] = 1; W_b[2][4] = 1; W_b[3][4] = 1
        
        U_b[0][5] = 1; U_b[3][5] = 1
        V_b[3][5] = 1
        W_b[0][5] = 1; W_b[1][5] = 1; W_b[2][5] = 1; W_b[3][5] = 1
        
        U_b[0][6] = 1; U_b[1][6] = 1
        V_b[1][6] = 1
        W_b[0][6] = 1; W_b[1][6] = 1; W_b[2][6] = 1; W_b[3][6] = 1
        
        W_b[0] = [1, 0, 0, 1, -1, 0, -1]
        W_b[1] = [-1, 0, 1, -1, 0, 0, 1]
        W_b[2] = [0, 1, -1, 0, 1, 0, -1]
        W_b[3] = [1, -1, 0, 1, 1, 0, 0]
        
        return U_b, V_b, W_b

    def schoolbook_block():
        U_b = [[0]*4 for _ in range(4)]
        V_b = [[0]*4 for _ in range(2)]
        W_b = [[0]*4 for _ in range(2)]
        
        for i in range(2):
            for j in range(2):
                col = i*2 + j
                U_b[i*2+j][col] = 1
                V_b[j][col] = 1
                W_b[j*2+i][col] = 1
        
        return U_b, V_b, W_b

    col = 0
    for block in range(5):
        U_b, V_b, W_b = strassen_block()
        for i in range(4):
            for c in range(7):
                U[i][col+c] = U_b[i][c]
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    V[j*11 + (2*block + k)][col+c] = V_b[j*2 + k][c]
        for k in range(2):
            for i in range(2):
                for c in range(7):
                    W[(2*block + k)*2 + i][col+c] = W_b[k*2 + i][c]
        col += 7
        
    U_b, V_b, W_b = schoolbook_block()
    for i in range(4):
        for c in range(4):
            U[i][col+c] = U_b[i][c]
    for j in range(2):
        for k in range(1):
            for c in range(4):
                V[j*11 + (10 + k)][col+c] = V_b[j*1 + k][c]
    for k in range(1):
        for i in range(2):
            for c in range(4):
                W[(10 + k)*2 + i][col+c] = W_b[k*2 + i][c]
    col += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
