DIMENSIONS = [2, 3, 4]

def construct():
    import time
    start = time.time()
    
    n, m, p = 2, 3, 4
    rank = 15
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Term 0: j=0, rank 1
    for i in range(2):
        U[i*3+0][0] = 1
    for k in range(4):
        V[0*4+k][0] = 1
    for k in range(4):
        for i in range(2):
            W[k*2+i][0] = 1
    
    strassen_U = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1},
    ]
    strassen_V = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1},
    ]
    strassen_W = [
        {(0,0):1, (1,1):1},
        {(0,1):1},
        {(1,0):1},
        {(0,0):-1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):1},
        {(0,0):-1, (1,1):-1},
    ]
    
    for r in range(7):
        col = 1 + r
        for (ip, jp), val in strassen_U[r].items():
            U[ip*3 + (jp+1)][col] = val
        for (jp, kp), val in strassen_V[r].items():
            V[(jp+1)*4 + kp][col] = val
        for (kp, ip), val in strassen_W[r].items():
            W[kp*2 + ip][col] = val
    
    for r in range(7):
        col = 8 + r
        for (ip, jp), val in strassen_U[r].items():
            U[ip*3 + (jp+1)][col] = val
        for (jp, kp), val in strassen_V[r].items():
            V[(jp+1)*4 + (kp+2)][col] = val
        for (kp, ip), val in strassen_W[r].items():
            W[(kp+2)*2 + ip][col] = val
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    # Fallback to schoolbook
                    rank24 = n*m*p
                    U2 = [[0]*rank24 for _ in range(n*m)]
                    V2 = [[0]*rank24 for _ in range(m*p)]
                    W2 = [[0]*rank24 for _ in range(p*n)]
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U2[i2*m+j2][col] = 1
                                V2[j2*p+k2][col] = 1
                                W2[k2*n+i2][col] = 1
                                col += 1
                    return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
