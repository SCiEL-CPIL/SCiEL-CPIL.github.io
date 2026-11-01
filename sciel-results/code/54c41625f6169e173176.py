import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # P1 = (a+d)(e+h)
    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    W[2][2] = 1
    W[3][2] = 1
    
    # P4 = d(g-f)
    U[3][3] = 1
    V[2][3] = 1
    V[1][3] = -1
    W[0][3] = 1
    W[1][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
