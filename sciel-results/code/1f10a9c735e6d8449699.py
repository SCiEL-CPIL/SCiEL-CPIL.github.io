import time

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Uses the schoolbook algorithm which has rank n*m*p = 3*3*7 = 63.
    The entries are integers (1 or 0), which satisfies the integer/half-integer constraint.
    """
    n, m, p = 3, 3, 7
    rank = n * m * p  # 63
    
    # Initialize matrices with zeros
    # U: (n*m) x rank = 9 x 63
    # V: (m*p) x rank = 21 x 63
    # W: (p*n) x rank = 21 x 63
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (Trace convention W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
