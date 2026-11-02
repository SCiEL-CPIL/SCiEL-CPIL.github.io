def construct():
    n, m, p = 4, 4, 7
    
    def get_strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[1][3]=1; W[3][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[2][4]=1; W[3][4]=-1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        
        return U, V, W
    
    def build_4x4x4():
        s2U, s2V, s2W = get_strassen_2x2x2()
        rank = 49
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(16)]
        
        block_terms = {
            0: {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'C': [(0,0,1), (1,1,1)]},
            1: {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)], 'C': [(1,0,1)]},
            2: {'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)], 'C': [(0,1,1)]},
            3: {'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)], 'C': [(1,0,1), (1,1,1)]},
            4: {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)], 'C': [(0,1,1), (1,1,-1)]},
            5: {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'C': [(1,1,1)]},
            6: {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)], 'C': [(0,0,1)]},
        }
        
        for t in range(7):
            info = block_terms[t]
            for s in range(7):
                r = t * 7 + s
                for (a, b, ca) in info['A']:
                    for i2 in range(2):
                        for j2 in range(2):
                            val = s2U[i2*2+j2][s]
                            if val != 0:
                                row = (2*a + i2) * 4 + (2*b + j2)
                                U[row][r] += ca * val
                for (b, c, cb) in info['B']:
                    for j2 in range(2):
                        for k2 in range(2):
                            val = s2V[j2*2+k2][s]
                            if val != 0:
                                row = (2*b + j2) * 4 + (2*c + k2)
                                V[row][r] += cb * val
                for (a, c, cc) in info['C']:
                    for k2 in range(2):
                        for i2 in range(2):
                            val = s2W[k2*2+i2][s]
                            if val != 0:
                                row = (2*c + k2) * 4 + (2*a + i2)
                                W[row][r] += cc * val
        return U, V, W
    
    U4, V4, W4 = build_4x4x4()
    
    U3 = [[0]*48 for _ in range(16)]
    V3 = [[0]*48 for _ in range(12)]
    W3 = [[0]*48 for _ in range(12)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U3[i*4+j][col] = 1
                V3[j*4+k][col] = 1
                W3[k*4+i][col] = 1
                col += 1
    
    total_rank = 49 + 48
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    for r in range(49):
        for row in range(16):
            U[row][r] = U4[row][r]
            j = row // 4
            k = row % 4
            V[j*7+k][r] = V4[row][r]
            k2 = row // 4
            i2 = row % 4
            W[k2*4+i2][r] = W4[row][r]
    
    for r in range(48):
        ro = 49 + r
        for row in range(16):
            U[row][ro] = U3[row][r]
            j = row // 4
            k = row % 4
            V[j*7+(4+k)][ro] = V3[row][r]
            k2 = row // 4
            i2 = row % 4
            W[(4+k2)*4+i2][ro] = W3[row][r]
    
    return {"U": U, "V": V, "W": W}
