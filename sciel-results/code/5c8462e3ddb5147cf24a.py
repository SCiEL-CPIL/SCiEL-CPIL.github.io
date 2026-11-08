import numpy as np

def construct():
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    r = 0
    
    # Part 1: j=2, all k (A[:,2] * B[2,:])
    for i in range(n):
        for k in range(p):
            U[i*m + 2, r] = 1
            V[2*p + k, r] = 1
            W[k*n + i, r] = 1
            r += 1
    
    # Part 2: j=0,1, k=0,1 (2x2x2 Strassen)
    # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[5], B[1,1]=V[6]
    # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    
    # M1 = (A00+A11)(B00+B11)
    U[0, r] = 1; U[4, r] = 1
    V[0, r] = 1; V[6, r] = 1
    W[0, r] = 1; W[3, r] = 1
    r += 1
    
    # M2 = (A10+A11)B00
    U[3, r] = 1; U[4, r] = 1
    V[0, r] = 1
    W[2, r] = 1; W[3, r] = -1
    r += 1
    
    # M3 = A00(B11-B00)
    U[0, r] = 1
    V[6, r] = 1; V[0, r] = -1
    W[1, r] = 1; W[3, r] = 1
    r += 1
    
    # M4 = A11(B00-B10)
    U[4, r] = 1
    V[0, r] = 1; V[5, r] = -1
    W[0, r] = 1; W[1, r] = 1
    r += 1
    
    # M5 = (A00+A01)B11
    U[0, r] = 1; U[1, r] = 1
    V[6, r] = 1
    W[0, r] = -1; W[1, r] = 1
    r += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3, r] = 1; U[0, r] = -1
    V[0, r] = 1; V[1, r] = 1
    W[2, r] = 1; W[3, r] = 1
    r += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, r] = 1; U[4, r] = -1
    V[5, r] = 1; V[6, r] = 1
    W[0, r] = 1
    r += 1
    
    # Part 3: j=0,1, k=2,3 (2x2x2 Strassen)
    # B[0,2]=V[2], B[0,3]=V[3], B[1,2]=V[7], B[1,3]=V[8]
    # C[0,2]=W[4], C[0,3]=W[5], C[1,2]=W[6], C[1,3]=W[7]
    
    # M1
    U[0, r] = 1; U[4, r] = 1
    V[2, r] = 1; V[8, r] = 1
    W[4, r] = 1; W[7, r] = 1
    r += 1
    
    # M2
    U[3, r] = 1; U[4, r] = 1
    V[2, r] = 1
    W[6, r] = 1; W[7, r] = -1
    r += 1
    
    # M3
    U[0, r] = 1
    V[8, r] = 1; V[2, r] = -1
    W[5, r] = 1; W[7, r] = 1
    r += 1
    
    # M4
    U[4, r] = 1
    V[2, r] = 1; V[7, r] = -1
    W[4, r] = 1; W[5, r] = 1
    r += 1
    
    # M5
    U[0, r] = 1; U[1, r] = 1
    V[8, r] = 1
    W[4, r] = -1; W[5, r] = 1
    r += 1
    
    # M6
    U[3, r] = 1; U[0, r] = -1
    V[2, r] = 1; V[3, r] = 1
    W[6, r] = 1; W[7, r] = 1
    r += 1
    
    # M7
    U[1, r] = 1; U[4, r] = -1
    V[7, r] = 1; V[8, r] = 1
    W[4, r] = 1
    r += 1
    
    # Part 4: j=0,1, k=4 (2x2x1)
    for i in range(n):
        for j in range(2):
            U[i*m + j, r] = 1
            V[j*p + 4, r] = 1
            W[4*n + i, r] = 1
            r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
