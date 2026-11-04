"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen coefficients for 2x2x2
    # U[i, j, c], V[j, k, c], W[k, i, c]
    # c from 0 to 6
    
    # M1 = (A00 + A11)(B00 + B11)
    # C00 += 1, C11 += 1
    U[0*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*3+0][col] = 1; V[1*3+1][col] = 1
    W[0*2+0][col] = 1; W[1*2+1][col] = 1
    col += 1
    
    # M2 = (A10 + A11)B00
    # C10 += 1
    U[1*3+0][col] = 1; U[1*3+1][col] = 1
    V[0*3+0][col] = 1
    W[0*2+1][col] = 1
    col += 1
    
    # M3 = A00(B01 - B11)
    # C01 += 1
    U[0*3+0][col] = 1
    V[0*3+1][col] = 1; V[1*3+1][col] = -1
    W[1*2+0][col] = 1
    col += 1
    
    # M4 = A11(B10 - B00)
    # C00 += 1, C10 += 1
    U[1*3+1][col] = 1
    V[1*3+0][col] = 1; V[0*3+0][col] = -1
    W[0*2+0][col] = 1; W[0*2+1][col] = 1
    col += 1
    
    # M5 = (A00 + A10)B11
    # C00 += -1, C01 += 1
    U[0*3+0][col] = 1; U[1*3+0][col] = 1
    V[1*3+1][col] = 1
    W[0*2+0][col] = -1; W[1*2+0][col] = 1
    col += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    # C11 += 1
    U[1*3+0][col] = 1; U[0*3+0][col] = -1
    V[0*3+0][col] = 1; V[0*3+1][col] = 1
    W[1*2+1][col] = 1
    col += 1
    
    # M7 = (A11 - A10)(B10 + B11)
    # C00 += 1
    U[1*3+1][col] = 1; U[1*3+0][col] = -1
    V[1*3+0][col] = 1; V[1*3+1][col] = 1
    W[0*2+0][col] = 1
    col += 1
    
    # Part 2: Schoolbook for i in {0,1}, j in {0,1}, k=2
    # Terms: (0,0,2), (0,1,2), (1,0,2), (1,1,2)
    for i in range(2):
        for j in range(2):
            k = 2
            U[i*3+j][col] = 1
            V[j*3+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    # Part 3: Schoolbook for i in {0,1}, j=2, k in {0,1,2}
    # Terms: (0,2,0), (0,2,1), (0,2,2), (1,2,0), (1,2,1), (1,2,2)
    for i in range(2):
        for k in range(3):
            j = 2
            U[i*3+j][col] = 1
            V[j*3+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
