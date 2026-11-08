import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        c_232 = c // 5
        c_115 = c % 5
        
        c_A = c_232 // 2
        c_B = c_232 % 2
        
        # U[i*m + j, c] = 1 if c_A == j
        for i in range(n):
            for j in range(m):
                if c_A == j:
                    U[i * m + j][c] = 1
                    
        # V[j*p + k, c] = 1 if c_A == j and c_B == k//5 and c_115 == k%5
        for j in range(m):
            for k in range(p):
                if c_A == j and c_B == k // 5 and c_115 == k % 5:
                    V[j * p + k][c] = 1
                    
        # W[k*n + i, c] = 1 if c_B == k//5 and c_115 == k%5
        for k in range(p):
            for i in range(n):
                if c_B == k // 5 and c_115 == k % 5:
                    W[k * n + i][c] = 1
                    
    return {"U": U, "V": V, "W": W}
