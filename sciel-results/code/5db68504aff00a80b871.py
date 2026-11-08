import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Strassen decomposition for (2x2)*(2x2)
    # T[i,j,k] = sum_r u[r][i*2+j] * v[r][j*2+k] * w[r][i*2+k] should equal 1
    strassen_u = [
        [1, 0, 1, 0],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1], # M7
    ]
    
    strassen_v = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1],  # M7
    ]
    
    strassen_w = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 1, 0, 1],   # M5
        [1, 0, 0, 0],   # M6
        [1, 0, 0, 0],   # M7
    ]
    
    # Verify Strassen decomposition
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for r in range(7):
                    total += strassen_u[r][i*2+j] * strassen_v[r][j*2+k] * strassen_w[r][i*2+k]
                assert total == 1, f"Strassen verification failed at ({i},{j},{k}): got {total}"
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for side in range(2):
        for q in range(4):
            for r in range(7):
                for i in range(2):
                    for jp in range(2):
                        j = side * 2 + jp
                        U[i * m + j][col] = strassen_u[r][i * 2 + jp]
                
                for jp in range(2):
                    j = side * 2 + jp
                    for kp in range(2):
                        k = q * 2 + kp
                        V[j * p + k][col] = strassen_v[r][jp * 2 + kp]
                
                for i in range(2):
                    for kp in range(2):
                        k = q * 2 + kp
                        W[k * n + i][col] = strassen_w[r][i * 2 + kp]
                
                col += 1
    
    # Verify full decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                assert abs(total - 1.0) < 1e-10, f"Decomposition failed at ({i},{j},{k}): got {total}"
    
    return {"U": U, "V": V, "W": W}
