import time

def construct():
    """
    Construct a rank-minimal (or best found) tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W'.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    """
    n, m, p = 2, 5, 6
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Populate the schoolbook decomposition
    # Each rank term r corresponds to a specific (i, j, k)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (trace convention: W[k*n+i, r])
                
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
    return {'U': U, 'V': V, 'W': W}
