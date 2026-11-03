import numpy as np
import time

def _strassen_2x2_factors():
    # Same as before
    pass

def _schoolbook_2x2x1_factors():
    # M_{2,2,1}
    # A (2x2), B (2x1), C (2x1)
    # C00 = A00 B00 + A01 B10
    # C10 = A10 B00 + A11 B10
    # Rank 4
    n, m, p = 2, 2, 1
    rank = 4
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    # Prods:
    # 1: A00 B00 -> C00
    # 2: A01 B10 -> C00
    # 3: A10 B00 -> C10
    # 4: A11 B10 -> C10
    
    # Indices:
    # U[i*m+j]
    # V[j*p+k] -> V[j]
    # W[k*n+i] -> W[i]
    
    # P1
    U[0*2+0][0] = 1.0
    V[0*1+0][0] = 1.0
    W[0*2+0][0] = 1.0
    
    # P2
    U[0*2+1][1] = 1.0
    V[1*1+0][1] = 1.0
    W[0*2+0][1] = 1.0
    
    # P3
    U[1*2+0][2] = 1.0
    V[0*1+0][2] = 1.0
    W[0*2+1][2] = 1.0
    
    # P4
    U[1*2+1][3] = 1.0
    V[1*1+0][3] = 1.0
    W[0*2+1][3] = 1.0
    
    return U, V, W

def _kron(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    R1 = len(W1[0])
    R2 = len(W2[0])
    R = R1 * R2
    n, m, p = n1*n2, m1*m2, p1*p2
    
    U = [[0.0]*R for _ in range(n*m)]
    V = [[0.0]*R for _ in range(m*p)]
    W = [[0.0]*R for _ in range(p*n)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for c1 in range(R1):
                v1 = U1[i1*m1+j1][c1]
                if v1 == 0: continue
                for i2 in range(n2):
                    for j2 in range(m2):
                        for c2 in range(R2):
                            v2 = U2[i2*m2+j2][c2]
                            if v2 == 0: continue
                            val = v1 * v2
                            i = i1*n2 + i2
                            j = j1*m2 + j2
                            c = c1*R2 + c2
                            U[i*m+j][c] += val
                            
    for j1 in range(m1):
        for k1 in range(p1):
            for c1 in range(R1):
                v1 = V1[j1*p1+k1][c1]
                if v1 == 0: continue
                for j2 in range(m2):
                    for k2 in range(p2):
                        for c2 in range(R2):
                            v2 = V2[j2*p2+k2][c2]
                            if v2 == 0: continue
                            val = v1 * v2
                            j = j1*m2 + j2
                            k = k1*p2 + k2
                            c = c1*R2 + c2
                            V[j*p+k][c] += val
                            
    for k1 in range(p1):
        for i1 in range(n1):
            for c1 in range(R1):
                v1 = W1[k1*n1+i1][c1]
                if v1 == 0: continue
                for k2 in range(p2):
                    for i2 in range(n2):
                        for c2 in range(R2):
                            v2 = W2[k2*n2+i2][c2]
                            if v2 == 0: continue
                            val = v1 * v2
                            k = k1*p2 + k2
                            i = i1*n2 + i2
                            c = c1*R2 + c2
                            W[k*n+i][c] += val
                            
    return U, V, W

def construct():
    n, m, p = 4, 4, 6
    
    # Get base factors
    U_s, V_s, W_s = _strassen_2x2_factors() # Rank 7
    U_sb, V_sb, W_sb = _schoolbook_2x2x1_factors() # Rank 4
    
    # M_{4,4,4} = M_{2,2,2} x M_{2,2,2}
    U_444, V_444, W_444 = _kron(U_s, V_s, W_s, U_s, V_s, W_s, 2, 2, 2, 2, 2, 2)
    
    # M_{4,4,2} = M_{2,2,1} x M_{2,2,2}
    U_442, V_442, W_442 = _kron(U_sb, V_sb, W_sb, U_s, V_s, W_s, 2, 2, 1, 2, 2, 2)
    
    # Concatenate
    R1 = len(W_444[0])
    R2 = len(W_442[0])
    R = R1 + R2
    
    U = [[0.0]*R for _ in range(n*m)]
    V = [[0.0]*R for _ in range(m*p)]
    W = [[0.0]*R for _ in range(p*n)]
    
    for i in range(n*m):
        for c in range(R1):
            U[i][c] = U_444[i][c]
        for c in range(R2):
            U[i][R1+c] = U_442[i][c]
            
    for i in range(m*p):
        for c in range(R1):
            V[i][c] = V_444[i][c]
        for c in range(R2):
            V[i][R1+c] = V_442[i][c]
            
    for i in range(p*n):
        for c in range(R1):
            W[i][c] = W_444[i][c]
        for c in range(R2):
            W[i][R1+c] = W_442[i][c]
            
    return {"U": U, "V": V, "W": W}
