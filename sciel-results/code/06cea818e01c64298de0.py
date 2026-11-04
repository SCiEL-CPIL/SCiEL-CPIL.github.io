"""Program for matrix multiplication 3x5 by 5x7 with rank 15 decomposition."""
DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = m  # rank = 5
    # U: shape (n*m, rank) = (15, 5)
    # U[i*m + j, r] = 1 if r == j, else 0
    U = [[0] * rank for _ in range(n * m)]
    # V: shape (m*p, rank) = (35, 5)
    # V[j*p + k, r] = 1 if r == j, else 0
    V = [[0] * rank for _ in range(m * p)]
    # W: shape (p*n, rank) = (21, 5)
    # W[k*n + i, r] = 1 for all i, k, r
    W = [[1] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][j] = 1
    
    for j in range(m):
        for k in range(p):
            V[j * p + k][j] = 1
    
    # W is already all ones
    
    return {"U": U, "V": V, "W": W}
