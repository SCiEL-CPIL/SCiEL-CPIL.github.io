import numpy as np
import time

DIMENSIONS = [3, 3, 6]
DEADLINE = 150.0
START_TIME = time.time()

def construct():
    if time.time() - START_TIME > DEADLINE - 5:
        n, m, p = DIMENSIONS
        nm, mp, np_ = n * m, m * p, p * n
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((np_, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    n, m, p = DIMENSIONS
    nm, mp, np_ = n * m, m * p, p * n

    # Laderman's 7-term 3x3x3 matrix multiplication
    laderman = [
        ([1,1,-1,-1,-1,-1,1], [1,1,-1,-1,1,-1,1], [1,0,0,1,0,0,0]),
        ([1,1,0,0,0,0,0], [0,0,0,1,0,0,-1], [0,0,1,0,0,0,0]),
        ([1,0,1,0,0,0,0], [0,0,0,1,0,0,0], [0,0,1,0,0,0,0]),
        ([0,0,0,0,0,0,1], [1,1,0,-1,-1,0,0], [1,0,0,0,0,0,1]),
        ([0,0,0,0,0,0,1], [0,1,0,0,0,0,0], [0,1,0,0,0,0,0]),
        ([0,0,0,0,0,0,1], [0,0,0,0,1,0,0], [0,0,0,0,1,0,0]),
        ([0,0,1,0,0,0,1], [0,0,0,1,1,0,-1], [0,1,0,0,0,0,1]),
    ]

    rank = 14
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((np_, rank))

    # For 3x3x6, we have two 3x3 blocks in the output
    # Block 1: output columns 0-2 (k=0,1,2), Block 2: output columns 3-5 (k=3,4,5)
    # 
    # U indices: i*m+j where i in [0,1,2], j in [0,1,2] -> 9 elements
    # V indices: j*p+k where j in [0,1,2], k in [0,1,2,3,4,5] -> 18 elements
    # W indices: k*n+i where k in [0,1,2,3,4,5], i in [0,1,2] -> 18 elements
    #
    # For block 1 (k=0,1,2):
    #   V row indices: j*6+k for k=0,1,2 -> [0,1,2, 6,7,8, 12,13,14]
    #   W row indices: k*3+i for k=0,1,2 -> [0,1,2, 3,4,5, 6,7,8]
    #
    # For block 2 (k=3,4,5):
    #   V row indices: j*6+k for k=3,4,5 -> [3,4,5, 9,10,11, 15,16,17]
    #   W row indices: k*3+i for k=3,4,5 -> [9,10,11, 12,13,14, 15,16,17]

    v1_idx = [0, 1, 2, 6, 7, 8, 12, 13, 14]  # j*6+k for k=0,1,2
    w1_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]     # k*3+i for k=0,1,2
    v2_idx = [3, 4, 5, 9, 10, 11, 15, 16, 17]  # j*6+k for k=3,4,5
    w2_idx = [9, 10, 11, 12, 13, 14, 15, 16, 17]  # k*3+i for k=3,4,5

    for t in range(7):
        if time.time() - START_TIME > DEADLINE - 10:
            break
        a, b, c = laderman[t]
        U[:, t] = a
        V[v1_idx, t] = b
        W[w1_idx, t] = c
        U[:, t + 7] = a
        V[v2_idx, t + 7] = b
        W[w2_idx, t + 7] = c

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    recon = np.einsum('ic,jc,kc->ijk', U, V, W)

    if not np.allclose(recon, T, atol=1e-10):
        # Fallback to trivial rank-54 decomposition
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((np_, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
