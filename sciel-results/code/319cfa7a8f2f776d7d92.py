import time

DIMENSIONS = [2, 4, 8]

def construct():
    # Start time guard
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Strassen 2x2x2 decomposition
    # U' shape (4, 7), V' shape (4, 7), W' shape (4, 7)
    U_prime = [[0]*7 for _ in range(4)]
    V_prime = [[0]*7 for _ in range(4)]
    W_prime = [[0]*7 for _ in range(4)]
    
    # r=0 (M1): A00+A11, B00+B11
    U_prime[0][0] = 1; U_prime[3][0] = 1
    V_prime[0][0] = 1; V_prime[3][0] = 1
    W_prime[0][0] = 1; W_prime[3][0] = 1 # C00, C11
    
    # r=1 (M2): A10+A11, B00
    U_prime[2][1] = 1; U_prime[3][1] = 1
    V_prime[0][1] = 1
    W_prime[1][1] = 1 # C10
    
    # r=2 (M3): A00, B01-B11
    U_prime[0][2] = 1
    V_prime[1][2] = 1; V_prime[3][2] = -1
    W_prime[2][2] = 1; W_prime[3][2] = 1 # C01, C11
    
    # r=3 (M4): A11, B10-B00
    U_prime[3][3] = 1
    V_prime[2][3] = 1; V_prime[0][3] = -1
    W_prime[0][3] = 1; W_prime[1][3] = 1 # C00, C10
    
    # r=4 (M5): A00+A10, B11
    U_prime[0][4] = 1; U_prime[2][4] = 1
    V_prime[3][4] = 1
    W_prime[0][4] = -1; W_prime[2][4] = 1 # C00, C01
    
    # r=5 (M6): A10-A00, B00+B01
    U_prime[2][5] = 1; U_prime[0][5] = -1
    V_prime[0][5] = 1; V_prime[1][5] = 1
    W_prime[3][5] = 1 # C11
    
    # r=6 (M7): A01-A11, B10+B11
    U_prime[1][6] = 1; U_prime[3][6] = -1
    V_prime[2][6] = 1; V_prime[3][6] = 1
    W_prime[0][6] = 1; W_prime[3][6] = -1 # C00, C11

    # Schoolbook 1x2x4 decomposition
    # U'' shape (2, 8), V'' shape (8, 8), W'' shape (4, 8)
    # Wait, indices for 1x2x4:
    # i'' in {0}, j'' in {0,1}, k'' in {0,1,2,3}
    # U'' index: i''*2 + j'' -> 0*2+j'' = j''
    # V'' index: j''*4 + k''
    # W'' index: k''*1 + 0 = k''
    
    U_double_prime = [[0]*8 for _ in range(2)]
    V_double_prime = [[0]*8 for _ in range(8)]
    W_double_prime = [[0]*8 for _ in range(4)]
    
    for r2 in range(8):
        j2 = r2 % 2
        k2 = r2 // 2
        
        U_double_prime[j2][r2] = 1
        V_double_prime[j2*4 + k2][r2] = 1
        W_double_prime[k2][r2] = 1

    # Combine
    rank = 7 * 8
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for r1 in range(7):
        for r2 in range(8):
            r = r1 * 8 + r2
            
            # U indices
            for i in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        j = 2*j1 + j2
                        u_val = U_prime[i*2 + j1][r1] * U_double_prime[j2][r2]
                        if u_val != 0:
                            U[i*4 + j][r] = u_val
            
            # V indices
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(4):
                            j = 2*j1 + j2
                            k = 2*k1 + k2
                            v_val = V_prime[j1*2 + k1][r1] * V_double_prime[j2*4 + k2][r2]
                            if v_val != 0:
                                V[j*8 + k][r] = v_val
            
            # W indices
            for k1 in range(2):
                for k2 in range(4):
                    for i1 in range(2):
                        k = 2*k1 + k2
                        i = i1
                        w_val = W_prime[k1*2 + i1][r1] * W_double_prime[k2][r2]
                        if w_val != 0:
                            W[k*2 + i][r] = w_val

    if time.time() > deadline:
        # Fallback to schoolbook if time runs out (unlikely for this small size)
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
