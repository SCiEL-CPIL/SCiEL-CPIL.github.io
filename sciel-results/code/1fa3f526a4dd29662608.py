import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # This is the standard rank for matrix multiplication M_{n,m,p}.
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is (n*m) x rank
                # Index in U: i*m + j
                U[i * m + j][column] = 1
                
                # V is (m*p) x rank
                # Index in V: j*p + k
                V[j * p + k][column] = 1
                
                # W is (p*n) x rank
                # Index in W: k*n + i (Trace convention: W[k*n+i, r])
                W[k * n + i][column] = 1
                
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
