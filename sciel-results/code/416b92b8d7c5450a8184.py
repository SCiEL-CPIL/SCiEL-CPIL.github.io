import time

N = 3
M = 4
P = 6

def construct():
    n, m, p = N, M, P
    rank = 12
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: j in {0, 1}
    # Ranks 0-5
    for i in range(n):
        for j_local in range(2):
            r = i * 2 + j_local
            j_global = j_local # 0 or 1
            
            # U: A[i, j_global]
            U[i * m + j_global][r] = 1
            
            # V: B[j_global, k] for all k
            for k in range(p):
                V[j_global * p + k][r] = 1
                
            # W: C[i, k]
            for k in range(p):
                W[k * n + i][r] = 1
                
    # Block 2: j in {2, 3}
    # Ranks 6-11
    for i in range(n):
        for j_local in range(2):
            r = 6 + i * 2 + j_local
            j_global = 2 + j_local # 2 or 3
            
            # U: A[i, j_global]
            U[i * m + j_global][r] = 1
            
            # V: B[j_global, k] for all k
            for k in range(p):
                V[j_global * p + k][r] = 1
                
            # W: C[i, k]
            for k in range(p):
                W[k * n + i][r] = 1

    return {'U': U, 'V': V, 'W': W}
