"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook + Strassen decomposition of rank n*m*p - 1: 
    # Replace a 2x2x2 sub-block (8 schoolbook terms) with Strassen (7 terms), saving 1 rank.
    n, m, p = DIMENSIONS
    rank = n * m * p - 1  # 59
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    
    # Strassen on sub-block i in {0,1}, j in {0,1}, k in {0,1}
    # U row = i*m + j, V row = j*p + k, W row = k*n + i
    # 7 Strassen terms:
    # m1: (U rows 0,0 & 1,1) x (V rows 0,0 & 1,1) -> W: (0,0)+(1,1)
    # m2: (U rows 1,0 & 1,1) x (V row 0,0) -> W: (0,1)+(1,0)
    # m3: (U row 0,0) x (V rows 0,1 & 1,1) -> W: (0,1)+(1,0)
    # m4: (U row 1,1) x (V rows 1,0 & 0,0) -> W: (0,1)+(1,0)
    # m5: (U rows 0,0 & 0,1) x (V row 1,1) -> W: (0,1)+(1,1)
    # m6: (U rows 1,0 & 0,0) x (V rows 0,0 & 0,1) -> W: (1,1)
    # m7: (U rows 0,1 & 1,1) x (V rows 1,0 & 1,1) -> W: (0,0)
    
    # m1 = (A00+A11)(B00+B11)
    U[0*m+0][column] = 1; U[1*m+1][column] = 1
    V[0*p+0][column] = 1; V[1*p+1][column] = 1
    W[0*n+0][column] = 1; W[1*n+1][column] = 1
    column += 1
    
    # m2 = (A10+A11)B00
    U[1*m+0][column] = 1; U[1*m+1][column] = 1
    V[0*p+0][column] = 1
    W[0*n+1][column] = 1; W[1*n+0][column] = 1
    column += 1
    
    # m3 = A00(B01-B11)
    U[0*m+0][column] = 1
    V[0*p+1][column] = 1; V[1*p+1][column] = -1
    W[0*n+1][column] = 1; W[1*n+0][column] = 1
    column += 1
    
    # m4 = A11(B10-B00)
    U[1*m+1][column] = 1
    V[1*p+0][column] = 1; V[0*p+0][column] = -1
    W[0*n+1][column] = 1; W[1*n+0][column] = 1
    column += 1
    
    # m5 = (A00+A01)B11
    U[0*m+0][column] = 1; U[0*m+1][column] = 1
    V[1*p+1][column] = 1
    W[0*n+1][column] = 1; W[1*n+1][column] = 1
    column += 1
    
    # m6 = (A10-A00)(B00+B01)
    U[1*m+0][column] = 1; U[0*m+0][column] = -1
    V[0*p+0][column] = 1; V[0*p+1][column] = 1
    W[1*n+1][column] = 1
    column += 1
    
    # m7 = (A01-A11)(B10+B11)
    U[0*m+1][column] = 1; U[1*m+1][column] = -1
    V[1*p+0][column] = 1; V[1*p+1][column] = 1
    W[0*n+0][column] = 1
    column += 1
    
    # Now fill in the remaining schoolbook terms for all (i,j,k) except those in the 2x2x2 block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already handled by Strassen
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    
    assert column == rank, f"Expected rank {rank}, got {column}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
