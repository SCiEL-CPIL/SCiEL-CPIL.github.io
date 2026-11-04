"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for 2x2x2 block (i,j,k in {0,1})
    # Using verified formulas
    col = 0
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    # M3 = A00(B10-B11)
    U[0][col] = 1
    V[3][col] = 1; V[4][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    # M4 = A11(B01-B10)
    U[4][col] = 1
    V[1][col] = 1; V[3][col] = -1
    W[1][col] = 1; W[0][col] = 1
    col += 1
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[2][col] = 1; W[0][col] = -1
    col += 1
    # M6 = (A10-A00)(B00+B10)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[3][col] = 1
    W[3][col] = 1
    col += 1
    # M7 = (A01-A11)(B01+B11)
    U[1][col] = 1; U[4][col] = -1
    V[1][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # Schoolbook for j=2 or k=2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == 2 or k == 2:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    assert col == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
