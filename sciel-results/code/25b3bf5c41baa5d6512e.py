import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    # Schoolbook rank: n*m*p = 72
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index for A[i,j]
                u_idx = i * m + j
                # V index for B[j,k]
                v_idx = j * p + k
                # W index for C[i,k] -> W[k*n + i]
                w_idx = k * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
