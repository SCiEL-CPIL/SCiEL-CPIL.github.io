# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 1 solution for T[i,j,k,l] = delta_{jk}
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1 if j==k else 0
    # W[k*n+l, 0] = 1
    
    R = 1
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Fill U
    for i in range(n):
        for j in range(m):
            U[i * m + j][0] = 1
            
    # Fill V
    for j in range(m):
        for k in range(m):
            idx = j * p + k
            if j == k:
                V[idx][0] = 1
            else:
                V[idx][0] = 0
                
    # Fill W
    for k in range(m):
        for l in range(p):
            idx = k * n + l
            W[idx][0] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
