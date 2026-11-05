def construct():
    n, m, p = DIMENSIONS
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for 2x2
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y11)
    # M4 = X11(Y01-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y10)
    # M7 = (X01-X11)(Y11+Y01)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4 + M5
    # C10 = M1 + M2 - M5 - M7
    # C11 = M2 + M3 - M1 + M6
    
    for c in range(4):
        # Top part: A[0:2, 0:2] * B[0:2, 2c:2c+2]
        # U indices for A[0:2, 0:2]: 0, 1, 4, 5
        # V indices for B[0:2, 2c:2c+2]: 2c, 2c+1, 8+2c, 9+2c
        # W indices for C[0:2, 2c:2c+2]: 6c, 6c+1, 6c+3, 6c+4
        
        # M1
        U[0][col] += 1; U[5][col] += 1
        V[2*c][col] += 1; V[8+2*c][col] += 1
        W[6*c][col] += 1; W[6*c+1][col] += 1
        col += 1
        
        # M2
        U[4][col] += 1; U[5][col] += 1
        V[2*c][col] += 1
        W[6*c+1][col] += 1; W[6*c+4][col] += 1
        col += 1
        
        # M3
        U[0][col] += 1
        V[8+2*c][col] += 1; V[9+2*c][col] -= 1
        W[6*c+3][col] += 1
        col += 1
        
        # M4
        U[5][col] += 1
        V[9+2*c][col] += 1; V[2*c][col] -= 1
        W[6*c][col] -= 1; W[6*c+3][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1; U[1][col] += 1
        V[9+2*c][col] += 1
        W[6*c][col] += 1; W[6*c+1][col] -= 1; W[6*c+3][col] += 1
        col += 1
        
        # M6
        U[4][col] += 1; U[0][col] -= 1
        V[2*c][col] += 1; V[8+2*c][col] += 1
        W[6*c+4][col] += 1
        col += 1
        
        # M7
        U[1][col] += 1; U[5][col] -= 1
        V[9+2*c][col] += 1; V[8+2*c][col] += 1
        W[6*c][col] += 1; W[6*c+1][col] -= 1
        col += 1
        
        # Second top part: A[0:2, 2:4] * B[2:4, 2c:2c+2]
        # U indices for A[0:2, 2:4]: 2, 3, 6, 7
        # V indices for B[2:4, 2c:2c+2]: 16+2c, 17+2c, 24+2c, 25+2c
        # W indices same
        
        # M1
        U[2][col] += 1; U[7][col] += 1
        V[16+2*c][col] += 1; V[24+2*c][col] += 1
        W[6*c][col] += 1; W[6*c+1][col] += 1
        col += 1
        
        # M2
        U[6][col] += 1; U[7][col] += 1
        V[16+2*c][col] += 1
        W[6*c+1][col] += 1; W[6*c+4][col] += 1
        col += 1
        
        # M3
        U[2][col] += 1
        V[24+2*c][col] += 1; V[25+2*c][col] -= 1
        W[6*c+3][col] += 1
        col += 1
        
        # M4
        U[7][col] += 1
        V[25+2*c][col] += 1; V[16+2*c][col] -= 1
        W[6*c][col] -= 1; W[6*c+3][col] += 1
        col += 1
        
        # M5
        U[2][col] += 1; U[3][col] += 1
        V[25+2*c][col] += 1
        W[6*c][col] += 1; W[6*c+1][col] -= 1; W[6*c+3][col] += 1
        col += 1
        
        # M6
        U[6][col] += 1; U[2][col] -= 1
        V[16+2*c][col] += 1; V[24+2*c][col] += 1
        W[6*c+4][col] += 1
        col += 1
        
        # M7
        U[3][col] += 1; U[7][col] -= 1
        V[25+2*c][col] += 1; V[24+2*c][col] += 1
        W[6*c][col] += 1; W[6*c+1][col] -= 1
        col += 1
        
        # Bottom part: A[2:3, 0:4] * B[0:4, 2c:2c+2]
        # U indices: 8, 9, 10, 11
        # V indices: 2c, 2c+1, 8+2c, 9+2c, 16+2c, 17+2c, 24+2c, 25+2c
        # W indices: 6c+2, 6c+5
        
        U[8][col] += 1; U[9][col] += 1; U[10][col] += 1; U[11][col] += 1
        V[2*c][col] += 1; V[2*c+1][col] += 1
        V[8+2*c][col] += 1; V[9+2*c][col] += 1
        V[16+2*c][col] += 1; V[17+2*c][col] += 1
        V[24+2*c][col] += 1; V[25+2*c][col] += 1
        W[6*c+2][col] += 1; W[6*c+5][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
