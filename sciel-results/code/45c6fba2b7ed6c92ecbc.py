import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def strassen_2x2():
    U = np.array([
        [1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 0, 1]
    ], dtype=int)
    V = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ], dtype=int)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    return U, V, W

def schoolbook(m, n, p):
    rank = m * n * p
    U = np.zeros((m * n, rank), dtype=int)
    V = np.zeros((n * p, rank), dtype=int)
    W = np.zeros((p * m, rank), dtype=int)
    col = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                U[i * n + j, col] = 1
                V[j * p + k, col] = 1
                W[k * m + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    
    # Strassen 2x2
    U2, V2, W2 = strassen_2x2()
    
    # Kronecker product for 4x4 @ 4x4
    U4 = np.kron(U2, U2)  # 16x49
    V4 = np.kron(V2, V2)  # 16x49
    W4 = np.kron(W2, W2)  # 16x49
    
    # Schoolbook for 4x4 @ 4x1
    U1, V1, W1 = schoolbook(4, 4, 1)  # 16x16, 4x16, 4x16
    
    # Combine for 4x4 @ 4x9
    rank = 49 + 49 + 16  # 114
    
    U_total = np.zeros((16, rank), dtype=int)
    U_total[:, 0:49] = U4
    U_total[:, 49:98] = U4
    U_total[:, 98:114] = U1
    
    V_total = np.zeros((36, rank), dtype=int)
    V_total[0:16, 0:49] = V4
    V_total[16:32, 49:98] = V4
    V_total[32:36, 98:114] = V1
    
    W_total = np.zeros((36, rank), dtype=int)
    W_total[0:16, 0:49] = W4
    W_total[16:32, 49:98] = W4
    W_total[32:36, 98:114] = W1
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
