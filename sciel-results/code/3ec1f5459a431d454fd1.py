"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=13
    # Rank 51 decomposition: Strassen (7) for first 2 cols + Schoolbook (44) for rest
    
    rank = 51
    U = [[0] * rank for _ in range(n * m)]  # 4 x 51
    V = [[0] * rank for _ in range(m * p)]  # 26 x 51
    W = [[0] * rank for _ in range(p * n)]  # 26 x 51
    
    # Indices for U (A matrix 2x2)
    # U[0]=a00, U[1]=a01, U[2]=a10, U[3]=a11
    
    # Indices for V (B matrix 2x13)
    # V[j*13 + k]
    # b00=V[0], b01=V[1], b02=V[2]...
    # b10=V[13], b11=V[14], b12=V[15]...
    
    # Indices for W (C matrix 2x13, flattened as W[k*n + i])
    # C is 2x13. C[i, k].
    # W index = k * 2 + i
    # c00 -> W[0], c10 -> W[1]
    # c01 -> W[2], c11 -> W[3]
    # c02 -> W[4], c12 -> W[5]
    # ...
    
    col = 0
    
    # Strassen terms for columns 0 and 1
    # M1 = (a00 + a11)(b00 + b11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[14][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = (a10 + a11)b00
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1
    
    # M3 = a00(b01 - b11)
    U[0][col] = 1
    V[1][col] = 1; V[14][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M4 = a11(b10 - b00)
    U[3][col] = 1
    V[13][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (a00 + a01)b11
    U[0][col] = 1; U[1][col] = 1
    V[14][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1
    
    # M6 = (a10 - a00)(b00 + b01)
    U[0][col] = -1; U[2][col] = 1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = -1
    col += 1
    
    # M7 = (a01 - a11)(b10 + b11)
    U[1][col] = 1; U[3][col] = -1
    V[13][col] = 1; V[14][col] = 1
    W[0][col] = 1
    col += 1
    
    # Schoolbook for columns 2 to 12
    for k in range(2, p):
        # C0k = a00*b0k + a01*b1k
        # Term 1: a00 * b0k -> C0k
        U[0][col] = 1
        V[k][col] = 1
        W[2*k][col] = 1
        col += 1
        
        # Term 2: a01 * b1k -> C0k
        U[1][col] = 1
        V[13+k][col] = 1
        W[2*k][col] = 1
        col += 1
        
        # C1k = a10*b0k + a11*b1k
        # Term 3: a10 * b0k -> C1k
        U[2][col] = 1
        V[k][col] = 1
        W[2*k+1][col] = 1
        col += 1
        
        # Term 4: a11 * b1k -> C1k
        U[3][col] = 1
        V[13+k][col] = 1
        W[2*k+1][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
