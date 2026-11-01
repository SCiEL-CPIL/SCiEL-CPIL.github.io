"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 59 decomposition
    # 7 (Strassen) + 16 (Schoolbook A_L B_T2) + 36 (Schoolbook A_R B_B)
    rank = 59
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen part for i in {0,1}, j in {0,1}, k in {0,1}
    # A indices: i*m+j
    # B indices: j*p+k
    # C indices: k*n+i
    
    # p1 = (A00 + A11)(B00 + B11)
    U[0][col] = 1; U[6][col] = 1
    V[0][col] = 1; V[7][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # p2 = (A10 + A11)B00
    U[5][col] = 1; U[6][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # p3 = A00(B10 - B00)
    U[0][col] = 1
    V[6][col] = 1; V[0][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # p4 = A11(B01 - B11)
    U[6][col] = 1
    V[1][col] = 1; V[7][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # p5 = (A00 + A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[0][col] = 1; W[2][col] = 1
    col += 1
    
    # p6 = (A10 - A00)(B00 + B01)
    U[5][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # p7 = (A01 - A11)(B10 + B11)
    U[1][col] = 1; U[6][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[0][col] = 1
    col += 1
    
    # Schoolbook part 2: A_L (i in 0,1; j in 0,1) * B_T2 (j in 0,1; k in 2,3,4,5)
    for i in range(2):
        for j in range(2):
            for k in range(2, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Schoolbook part 3: A_R (i in 0,1; j in 2,3,4) * B_B (j in 2,3,4; k in 0,1,2,3,4,5)
    for i in range(2):
        for j in range(2, 5):
            for k in range(6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
