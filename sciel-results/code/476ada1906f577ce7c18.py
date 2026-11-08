import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 118  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Strassen 2x2x2 decomposition
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ])
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ])
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ])

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
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    U1[row_U, col] = S_U[i1*2+j1, c1] * S_U[i2*2+j2, c2]

    for j in range(4):
        for k in range(4):
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            row_V = j * p + k
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    V1[row_V, col] = S_V[j1*2+k1, c1] * S_V[j2*2+k2, c2]

    for k in range(4):
        for i in range(4):
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            row_W = k * n + i
            for c1 in range(7):
                for c2 in range(7):
                    col = c1 * 7 + c2
                    W1[row_W, col] = S_W[k1*2+i1, c1] * S_W[k2*2+i2, c2]

    # Part 2: 4x4x2 decomposition with rank 28 (tiling)
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
                        k = 4 + dk  # k in {4, 5}
                        V2[j * p + k, col] = S_V[dj * 2 + dk, c]
                
                for dk in range(2):
                    for di in range(2):
                        k = 4 + dk
                        i = bi * 2 + di
                        W2[k * n + i, col] = S_W[dk * 2 + di, c]

    # Combine
    total_rank = rank1 + rank2
    U = np.concatenate([U1, U2], axis=1)
    V = np.concatenate([V1, V2], axis=1)
    W = np.concatenate([W1, W2], axis=1)

    # Remove zero columns to reduce rank
    keep_cols = []
    for c in range(total_rank):
        if np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0):
            keep_cols.append(c)

    U_final = U[:, keep_cols]
    V_final = V[:, keep_cols]
    W_final = W[:, keep_cols]

    # Convert to lists of lists for the return format
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
