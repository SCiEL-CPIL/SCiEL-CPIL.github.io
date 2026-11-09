import numpy as np

def construct():
    # 3x3 Schoolbook
    n1, m1, p1 = 3, 3, 3
    r1 = 27
    U1 = [[0]*r1 for _ in range(n1*m1)]
    V1 = [[0]*r1 for _ in range(m1*p1)]
    W1 = [[0]*r1 for _ in range(p1*n1)]
    
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                c = i*p1*p1 + j*p1 + k
                U1[i*m1+j][c] = 1
                V1[j*p1+k][c] = 1
                W1[k*n1+i][c] = 1
                
    # 2x2 Strassen
    n2, m2, p2 = 2, 2, 2
    r2 = 7
    U2 = [[0]*r2 for _ in range(n2*m2)]
    V2 = [[0]*r2 for _ in range(m2*p2)]
    W2 = [[0]*r2 for _ in range(p2*n2)]
    
    # M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[3][1] = 1
    
    # M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1
    
    # M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[3][3] = 1
    
    # M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Kronecker Product
    n, m, p = n1*n2, m1*m2, p1*p2
    r = r1*r2
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1*r2 + c2
            for i1 in range(n1):
                for i2 in range(n2):
                    i = i1*n2 + i2
                    for j1 in range(m1):
                        for j2 in range(m2):
                            j = j1*m2 + j2
                            u1 = U1[i1*m1+j1][c1]
                            u2 = U2[i2*m2+j2][c2]
                            if u1 != 0 and u2 != 0:
                                U[i*m+j][c] = u1 * u2
                                
            for j1 in range(m1):
                for j2 in range(m2):
                    j = j1*m2 + j2
                    for k1 in range(p1):
                        for k2 in range(p2):
                            k = k1*p2 + k2
                            v1 = V1[j1*p1+k1][c1]
                            v2 = V2[j2*p2+k2][c2]
                            if v1 != 0 and v2 != 0:
                                V[j*p+k][c] = v1 * v2
                                
            for k1 in range(p1):
                for k2 in range(p2):
                    k = k1*p2 + k2
                    for i1 in range(n1):
                        for i2 in range(n2):
                            i = i1*n2 + i2
                            w1 = W1[k1*n1+i1][c1]
                            w2 = W2[k2*n2+i2][c2]
                            if w1 != 0 and w2 != 0:
                                W[k*n+i][c] = w1 * w2
                                
    return {"U": U, "V": V, "W": W}
