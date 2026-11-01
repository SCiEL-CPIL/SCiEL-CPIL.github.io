import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients
    # a_s[i, j] where i,j in {0,1}
    # b_s[j, k] where j,k in {0,1}
    # c_s[i, k] where i,k in {0,1}
    
    a_coeffs = [
        [[1, 0], [0, 1]], # 1
        [[0, 0], [1, 1]], # 2
        [[1, 0], [0, 0]], # 3
        [[0, 0], [0, 1]], # 4
        [[1, 1], [0, 0]], # 5
        [[-1, 0], [1, 0]], # 6
        [[0, 1], [0, -1]]  # 7
    ]
    
    b_coeffs = [
        [[1, 0], [0, 1]], # 1
        [[1, 0], [0, 0]], # 2
        [[0, 1], [0, -1]], # 3
        [[-1, 0], [1, 0]], # 4
        [[0, 0], [0, 1]], # 5
        [[1, 1], [0, 0]], # 6
        [[0, 0], [1, 1]]  # 7
    ]
    
    c_coeffs = [
        [[1, 0], [0, 1]], # 1
        [[0, 0], [1, -1]], # 2
        [[0, 1], [1, 1]], # 3
        [[1, 0], [1, 0]], # 4
        [[-1, 1], [0, 0]], # 5
        [[0, 0], [0, 1]], # 6
        [[1, 0], [0, 0]]  # 7
    ]
    
    col = 0
    for q in range(p // 2):
        for s in range(7):
            a_s = a_coeffs[s]
            b_s = b_coeffs[s]
            c_s = c_coeffs[s]
            
            for i in range(n):
                for j in range(m):
                    U[i*m + j, col] = a_s[i][j]
                    
            for j in range(m):
                for k_local in range(2):
                    k_global = 2*q + k_local
                    V[j*p + k_global, col] = b_s[j][k_local]
                    
            for k_local in range(2):
                k_global = 2*q + k_local
                for i in range(n):
                    W[k_global*n + i, col] = c_s[i][k_local]
                    
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
