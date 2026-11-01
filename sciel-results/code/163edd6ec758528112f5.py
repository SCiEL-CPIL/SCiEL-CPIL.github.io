import time

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    pairs = [(0,1), (2,3), (4,5), (6,7), (8,9)]
    single = 10
    
    rank = 7 * len(pairs) + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k1, k2 in pairs:
        # M1: (A00+A11)*(B[0,k1]+B[1,k2])
        U[0][col] = 1; U[3][col] = 1
        V[k1][col] = 1; V[p+k2][col] = 1
        W[2*k1][col] = 1; W[2*k2+1][col] = 1
        col += 1
        # M2: (A10+A11)*B[0,k1]
        U[2][col] = 1; U[3][col] = 1
        V[k1][col] = 1
        W[2*k1+1][col] = 1
        col += 1
        # M3: A00*(B[1,k1]-B[1,k2])
        U[0][col] = 1
        V[p+k1][col] = 1; V[p+k2][col] = -1
        W[2*k2][col] = 1; W[2*k2+1][col] = 1
        col += 1
        # M4: A11*(B[0,k2]-B[0,k1])
        U[3][col] = 1
        V[k2][col] = 1; V[k1][col] = -1
        W[2*k1][col] = 1; W[2*k1+1][col] = 1
        col += 1
        # M5: (A00+A01)*B[1,k2]
        U[0][col] = 1; U[1][col] = 1
        V[p+k2][col] = 1
        W[2*k1][col] = -1; W[2*k2][col] = 1
        col += 1
        # M6: (A10-A00)*(B[0,k1]+B[1,k1])
        U[2][col] = 1; U[0][col] = -1
        V[k1][col] = 1; V[p+k1][col] = 1
        W[2*k2+1][col] = 1
        col += 1
        # M7: (A01-A11)*(B[0,k2]+B[1,k2])
        U[1][col] = 1; U[3][col] = -1
        V[k2][col] = 1; V[p+k2][col] = 1
        W[2*k1][col] = 1; W[2*k2+1][col] = 1
        col += 1
    
    k3 = single
    # Schoolbook for single column
    U[0][col] = 1; V[k3][col] = 1; W[2*k3][col] = 1; col += 1
    U[1][col] = 1; V[p+k3][col] = 1; W[2*k3][col] = 1; col += 1
    U[2][col] = 1; V[k3][col] = 1; W[2*k3+1][col] = 1; col += 1
    U[3][col] = 1; V[p+k3][col] = 1; W[2*k3+1][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
