import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
DEADLINE = 150.0

def construct():
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54

    # U: (n*m, rank) = (9, 54)
    # V: (m*p, rank) = (18, 54)
    # W: (p*n, rank) = (18, 54)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
