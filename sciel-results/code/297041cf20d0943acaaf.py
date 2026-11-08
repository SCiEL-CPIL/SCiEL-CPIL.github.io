import numpy as np

def construct():
    n, m, p = 2, 3, 5
    nm = n * m  # 6
    mp = m * p  # 15
    np_ = n * p  # 10
    
    # Schoolbook solution with rank n*m*p = 30
    R = n * m * p
    U = np.zeros((R, nm))
    V = np.zeros((R, mp))
    W = np.zeros((R, np_))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[col, i * m + j] = 1
                V[col, j * p + k] = 1
                W[col, k * n + i] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
