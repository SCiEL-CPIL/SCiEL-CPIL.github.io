import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 29 decomposition
    # 7 columns for Strassen on 2x2x2 sub-block (i in 0,1; j in 0,1; k in 0,1)
    # 22 columns for schoolbook on remaining terms
    rank = 29
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part (columns 0-6)
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[6][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10 + A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = A00(B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[6][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10 - B00)
    U[4][3] = 1
    V[5][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[1][3] = 1
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[6][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[4][6] = -1
    V[5][6] = 1; V[6][6] = 1
    W[0][6] = 1
    
    # Schoolbook part (columns 7-28)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the 2x2x2 block handled by Strassen
                if j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
