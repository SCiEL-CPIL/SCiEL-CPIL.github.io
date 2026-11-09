import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 115  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen 2x2x2 decomposition (rank 7)
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=int)
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ], dtype=int)
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=int)
    
    # Part 1: 4x4x4 decomposition with rank 49 (tensor product of two Strassen)
    rank1 = 49
    U1 = np.zeros((n*m, rank1), dtype=int)
    V1 = np.zeros((m*p, rank1), dtype=int)
    W1 = np.zeros((p*n, rank1), dtype=int)
    
    for i in range(4):
        for j in range(4):
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            row_U = i * m + j
            U1[row_U, :] = np.outer(S_U[i1*2+j1], S_U[i2*2+j2]).flatten()
    
    for j in range(4):
        for k in range(4):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            row_V = j * p + k
            V1[row_V, :] = np.outer(S_V[j1*2+k1], S_V[j2*2+k2]).flatten()
    
    for k in range(4):
        for i in range(4):
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            row_W = k * n + i
            W1[row_W, :] = np.outer(S_W[k1*2+i1], S_W[k2*2+i2]).flatten()
    
    # Part 2: 4x4x2 decomposition with rank 28 (tiling with Strassen)
    rank2 = 28
    U2 = np.zeros((n*m, rank2), dtype=int)
    V2 = np.zeros((m*p, rank2), dtype=int)
    W2 = np.zeros((p*n, rank2), dtype=int)
    
    for bi in range(2):
        for bj in range(2):
            for c in range(7):
                col = (bi * 2 + bj) * 7 + c
                for di in range(2):
                    for dj in range(2):
                        i = bi * 2 + di
                        j = bj * 2 + dj
                        U2[i * m + j, col] = S_U[di * 2 + dj, c]
                
                for dj in range(2):
                    for dk in range(2):
                        j = bj * 2 + dj
                        k = 4 + dk
                        V2[j * p + k, col] = S_V[dj * 2 + dk, c]
                
                for dk in range(2):
                    for di in range(2):
                        k = 4 + dk
                        i = bi * 2 + di
                        W2[k * n + i, col] = S_W[dk * 2 + di, c]
    
    # Combine
    total_rank = rank1 + rank2
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    U[:, :rank1] = U1
    U[:, rank1:] = U2
    V[:, :rank1] = V1
    V[:, rank1:] = V2
    W[:, :rank1] = W1
    W[:, rank1:] = W2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
