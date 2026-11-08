import numpy as np

def construct():
    n, m, p = 2, 4, 6
    
    # Strassen for 2x2x2
    n1, m1, p1 = 2, 2, 2
    r1 = 7
    U1 = np.zeros((n1*m1, r1))
    V1 = np.zeros((m1*p1, r1))
    W1 = np.zeros((p1*n1, r1))
    
    # M1: A00*B00 -> C00
    U1[0,0] = 1; V1[0,0] = 1; W1[0,0] = 1
    # M2: (A00+A11)*B11 -> C10(+1), C11(-1)
    U1[0,1] = 1; U1[3,1] = 1; V1[3,1] = 1; W1[1,1] = 1; W1[3,1] = -1
    # M3: A11*(B00+B11) -> C01(+1), C11(+1)
    U1[3,2] = 1; V1[0,2] = 1; V1[3,2] = 1; W1[1,2] = 1; W1[3,2] = 1
    # M4: A00*(B11-B01) -> C00(+1), C10(+1)
    U1[0,3] = 1; V1[3,3] = 1; V1[1,3] = -1; W1[0,3] = 1; W1[1,3] = 1
    # M5: (A11-A00)*(B00+B01) -> C00(-1), C01(+1)
    U1[3,4] = 1; U1[0,4] = -1; V1[0,4] = 1; V1[1,4] = 1; W1[0,4] = -1; W1[1,4] = 1
    # M6: (A01+A11)*(B10-B00) -> C11(+1)
    U1[1,5] = 1; U1[3,5] = 1; V1[2,5] = 1; V1[0,5] = -1; W1[3,5] = 1
    # M7: (A00-A10)*(B01+B11) -> C00(+1)
    U1[0,6] = 1; U1[2,6] = -1; V1[1,6] = 1; V1[3,6] = 1; W1[0,6] = 1
    
    # Trivial for 1x2x3
    n2, m2, p2 = 1, 2, 3
    r2 = 6
    U2 = np.zeros((n2*m2, r2))
    V2 = np.zeros((m2*p2, r2))
    W2 = np.zeros((p2*n2, r2))
    
    for j in range(m2):
        for k in range(p2):
            c = j*p2 + k
            U2[j, c] = 1
            V2[j*p2+k, c] = 1
            W2[k, c] = 1
    
    # Kronecker product with row permutations
    # U': shape (n*m, r1*r2) = (8, 42)
    # Row index in U': r' = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
    # where i1 = (a//m1), j1 = (a%m1), a = i1*m1+j1
    # and i2 = (b//m2), j2 = (b%m2), b = i2*m2+j2
    
    U = np.zeros((n*m, r1*r2))
    for a in range(n1*m1):
        i1 = a // m1
        j1 = a % m1
        for b in range(n2*m2):
            i2 = b // m2
            j2 = b % m2
            row = (i1*n2 + i2)*(m1*m2) + (j1*m2 + j2)
            for c1 in range(r1):
                for c2 in range(r2):
                    c = c1*r2 + c2
                    U[row, c] = U1[a, c1] * U2[b, c2]
    
    # V': shape (m*p, r1*r2) = (24, 42)
    # Row index: (j1*p1+k1)*(m2*p2) + (j2*p2+k2)
    V = np.zeros((m*p, r1*r2))
    for a in range(m1*p1):
        j1 = a // p1
        k1 = a % p1
        for b in range(m2*p2):
            j2 = b // p2
            k2 = b % p2
            row = (j1*p1 + k1)*(m2*p2) + (j2*p2 + k2)
            for c1 in range(r1):
                for c2 in range(r2):
                    c = c1*r2 + c2
                    V[row, c] = V1[a, c1] * V2[b, c2]
    
    # W': shape (p*n, r1*r2) = (12, 42)
    # Row index: (k1*n1+i1)*(p2*n2) + (k2*n2+i2)
    W = np.zeros((p*n, r1*r2))
    for a in range(p1*n1):
        k1 = a // n1
        i1 = a % n1
        for b in range(p2*n2):
            k2 = b // n2
            i2 = b % n2
            row = (k1*n1 + i1)*(p2*n2) + (k2*n2 + i2)
            for c1 in range(r1):
                for c2 in range(r2):
                    c = c1*r2 + c2
                    W[row, c] = W1[a, c1] * W2[b, c2]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
