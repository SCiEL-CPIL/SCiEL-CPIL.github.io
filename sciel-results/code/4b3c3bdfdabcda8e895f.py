import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    rank = 55  # 5 pairs * (7 Strassen + 4 schoolbook)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 2x2x2 decomposition (7 terms)
    # U_local[i*2+j], V_local[j*2+k], W_local[k*2+i]
    strassen_U = [
        [1, 0, 0, 1],  # r=0
        [0, 0, 1, 1],  # r=1
        [1, 0, 0, 0],  # r=2
        [0, 0, 0, 1],  # r=3
        [1, 1, 0, 0],  # r=4
        [0, -1, 1, 0], # r=5
        [0, 1, 0, -1]  # r=6
    ]
    strassen_V = [
        [1, 0, 0, 1],  # r=0
        [1, 0, 0, 0],  # r=1
        [0, 1, 0, -1], # r=2
        [0, 0, 1, -1], # r=3
        [0, 0, 0, 1],  # r=4
        [1, 1, 0, 0],  # r=5
        [0, 0, 1, 1]   # r=6
    ]
    strassen_W = [
        [1, 0, 0, 1],  # r=0
        [0, 0, 0, 1],  # r=1
        [0, 1, 0, 0],  # r=2
        [0, 0, 1, 0],  # r=3
        [0, 1, 0, 0],  # r=4
        [0, 0, 1, 1],  # r=5
        [1, 0, 1, 0]   # r=6
    ]
    
    col = 0
    for t in range(5):  # 5 column pairs
        # Strassen for 2x2x2 (A[:,0:2] * B[0:2, 2t:2t+2])
        for r in range(7):
            # U: A[i, j] for i in {0,1}, j in {0,1}
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = strassen_U[r][i * 2 + j]
            # V: B[j, k] for j in {0,1}, k in {0,1} → global k = 2t + k_local
            for j in range(2):
                for k in range(2):
                    V[j * p + (2 * t + k), col] = strassen_V[r][j * 2 + k]
            # W: C[i, k] for i in {0,1}, k in {0,1} → global k = 2t + k_local
            for i in range(2):
                for k in range(2):
                    W[(2 * t + k) * n + i, col] = strassen_W[r][k * 2 + i]
            col += 1
        
        # Schoolbook for 2x1x2 (A[:,2] * B[2, 2t:2t+2])
        # 4 terms
        for i in range(2):
            for k in range(2):
                U[i * m + 2, col] = 1.0  # A[i, 2]
                V[2 * p + (2 * t + k), col] = 1.0  # B[2, 2t+k]
                W[(2 * t + k) * n + i, col] = 1.0  # C[i, 2t+k]
                col += 1
    
    def to_list(mat):
        return mat.tolist()
    
    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }
