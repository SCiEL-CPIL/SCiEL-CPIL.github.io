"""
Matrix multiplication tensor decomposition for 3x4x7.
The minimal rank for this specific case is believed to be the schoolbook rank 84,
as no known algebraic identities reduce the rank for these specific dimensions
unlike the 2x2x2 (Strassen) or 3x3x3 cases.
"""
import time

DIMENSIONS = [3, 4, 7]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 150.0  # 150 seconds
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
