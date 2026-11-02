def construct():
    n, m, p = 2, 2, 16
    rank = 56  # 8 column-pairs * 7 Strassen products
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    col = 0
    for c in range(8):  # 8 column-pairs
        k0 = 2 * c
        k1 = 2 * c + 1
        
        # Strassen product 1: (A00+A11)(B[0,k0]+B[1,k0])
        # U: A00=U[0], A11=U[3]
        U[0][col] = 1; U[3][col] = 1
        # V: B[0,k0]=V[0*16+k0], B[1,k0]=V[1*16+k0]
        V[k0][col] = 1; V[16+k0][col] = 1
        # W: C[0,k0]=W[k0*2+0], C[1,k0]=W[k0*2+1], C[0,k1]=W[k1*2+0], C[1,k1]=W[k1*2+1]
        W[k0*2+0][col] = 1  # C[0,k0] += M1
        W[k0*2+1][col] = 1  # C[1,k0] += M1
        W[k1*2+1][col] = 1  # C[1,k1] += M1
        col += 1
        
        # Strassen product 2: (A10+A11)B[0,k0]
        U[2][col] = 1; U[3][col] = 1  # A10=U[2], A11=U[3]
        V[k0][col] = 1  # B[0,k0]
        W[k0*2+1][col] = 1  # C[1,k0] += M2
        W[k1*2+1][col] = -1  # C[1,k1] -= M2
        col += 1
        
        # Strassen product 3: A00(B[1,k0]-B[1,k1])
        U[0][col] = 1  # A00
        V[16+k0][col] = 1; V[16+k1][col] = -1  # B[1,k0]-B[1,k1]
        W[k1*2+0][col] = 1  # C[0,k1] += M3
        W[k1*2+1][col] = 1  # C[1,k1] += M3
        col += 1
        
        # Strassen product 4: A11(B[0,k0]-B[0,k1])
        U[3][col] = 1  # A11
        V[k0][col] = 1; V[k1][col] = -1  # B[0,k0]-B[0,k1]
        W[k0*2+0][col] = 1  # C[0,k0] += M4
        W[k0*2+1][col] = 1  # C[1,k0] += M4
        col += 1
        
        # Strassen product 5: (A00+A01)B[1,k1]
        U[0][col] = 1; U[1][col] = 1  # A00=U[0], A01=U[1]
        V[16+k1][col] = 1  # B[1,k1]
        W[k0*2+0][col] = -1  # C[0,k0] -= M5
        W[k1*2+0][col] = 1  # C[0,k1] += M5
        col += 1
        
        # Strassen product 6: (A10-A00)(B[0,k0]+B[0,k1])
        U[2][col] = 1; U[0][col] = -1  # A10-A00
        V[k0][col] = 1; V[k1][col] = 1  # B[0,k0]+B[0,k1]
        W[k1*2+1][col] = 1  # C[1,k1] += M6
        col += 1
        
        # Strassen product 7: (A01-A11)(B[1,k0]+B[1,k1])
        U[1][col] = 1; U[3][col] = -1  # A01-A11
        V[16+k0][col] = 1; V[16+k1][col] = 1  # B[1,k0]+B[1,k1]
        W[k0*2+0][col] = 1  # C[0,k0] += M7
        col += 1
    
    return {"U": U, "V": V, "W": W}
