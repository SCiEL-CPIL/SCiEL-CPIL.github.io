import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices for the standard schoolbook decomposition
    # Each term (i, j, k) corresponds to A[i,j] * B[j,k] contributing to C[k,i] (based on W index convention)
    # W index is k*n + i
    # U index is i*m + j
    # V index is j*p + k
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Term t uses U[i*m+j], V[j*p+k], W[k*n+i]
                U[i * m + j][t] = 1
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}
