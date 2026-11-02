DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Term 0-5: A B1 (schoolbook 2x3x1)
    for i in range(n):
        for j in range(m):
            t = i * m + j
            U[t][j] = 1
            V[j * p][j] = 1
            W[j * n + i][j] = 1
            # Wait, B1 is first column of B, so k=0.
            # V index is j*p + k = j*3 + 0.
            # W index is k*n + i = 0*2 + i = i.
            # So V[j*3][t] = 1, W[i][t] = 1.
            
    # Let's rewrite clearly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A B1 (2x3 * 3x1)
    # A is 2x3, B1 is 3x1.
    # U[i*3+j, t] = 1
    # V[j*3+0, t] = 1
    # W[0*2+i, t] = 1
    t = 0
    for i in range(n):
        for j in range(m):
            U[i*m+j][t] = 1
            V[j*p+0][t] = 1
            W[0*n+i][t] = 1
            t += 1
            
    # Part 2: A1 B21 (2x1 * 1x2)
    # A1 is first column of A (j=0).
    # B21 is second column of B (k=1).
    # U[i*3+0, t] = 1
    # V[0*3+1, t] = 1
    # W[1*2+i, t] = 1
    for i in range(n):
        U[i*m+0][t] = 1
        V[0*p+1][t] = 1
        W[1*n+i][t] = 1
        t += 1
        
    # Part 3: A2 B22 (2x2 * 2x2) Strassen
    # A2 is A[:, 1:3], B22 is B[1:3, 1:3].
    # P00=A[0,1], P01=A[0,2], P10=A[1,1], P11=A[1,2]
    # Q00=B[1,1], Q01=B[1,2], Q10=B[2,1], Q11=B[2,2]
    # C00=W[1,0], C01=W[2,0], C10=W[1,1], C11=W[2,1]
    
    # M1 = (P00+P11)(Q00+Q11)
    U[0*m+1][t] = 1
    U[1*m+2][t] = 1
    V[1*p+1][t] = 1
    V[2*p+2][t] = 1
    W[1*n+0][t] = 1
    W[2*n+1][t] = 1
    t += 1
    
    # M2 = (P10+P11)Q00
    U[1*m+1][t] = 1
    U[1*m+2][t] = 1
    V[1*p+1][t] = 1
    W[1*n+1][t] = 1
    W[2*n+1][t] = -1
    t += 1
    
    # M3 = P00(Q10-Q00)
    U[0*m+1][t] = 1
    V[2*p+1][t] = 1
    V[1*p+1][t] = -1
    W[2*n+0][t] = 1
    W[2*n+1][t] = -1
    t += 1
    
    # M4 = P11(Q01-Q10)
    U[1*m+2][t] = 1
    V[1*p+2][t] = 1
    V[2*p+1][t] = -1
    W[1*n+0][t] = -1
    W[2*n+0][t] = 1
    t += 1
    
    # M5 = (P00+P01)Q11
    U[0*m+1][t] = 1
    U[0*m+2][t] = 1
    V[2*p+2][t] = 1
    W[1*n+0][t] = 1
    W[2*n+1][t] = 1
    t += 1
    
    # M6 = (P10-P00)(Q00+Q01)
    U[1*m+1][t] = 1
    U[0*m+1][t] = -1
    V[1*p+1][t] = 1
    V[1*p+2][t] = 1
    W[1*n+1][t] = 1
    W[2*n+1][t] = 1
    t += 1
    
    # M7 = (P01-P11)(Q10+Q11)
    U[0*m+2][t] = 1
    U[1*m+2][t] = -1
    V[2*p+1][t] = 1
    V[2*p+2][t] = 1
    W[1*n+1][t] = 1
    W[2*n+1][t] = -1
    t += 1
    
    return {"U": U, "V": V, "W": W}
