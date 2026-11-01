"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition of rank 7
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U: i*m+j -> 0:a11, 1:a12, 2:a21, 3:a22
    # V: j*p+k -> 0:b11, 1:b12, 2:b21, 3:b22
    # W: k*n+i -> 0:C11, 1:C21, 2:C12, 3:C22
    
    # M1 = (a11 + a22)(b11 + b22) -> C11 + C22
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a21 + a22)b11 -> C21
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = a11(b12 - b22) -> -C11 + C12
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = -1; W[2][2] = 1
    
    # M4 = a22(b21 - b11) -> -C11 + C21
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[1][3] = 1
    
    # M5 = (a11 + a12)b22 -> -C11 + C12
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (a21 - a11)(b11 + b12) -> C22
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a12 - a22)(b21 + b22) -> -C11 + C22
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
