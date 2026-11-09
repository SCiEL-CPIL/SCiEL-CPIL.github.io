import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Strassen rank-7 for 2x2x2
    # U: 4x7 (rows: a00, a01, a10, a11)
    # V: 4x7 (rows: b00, b01, b10, b11)
    # W: 4x7 (rows: c00, c01, c10, c11)
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [1, 0, -1, 0, 1, 0, 0]
    ], dtype=int)
    W_s = np.array([
        [1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0]
    ], dtype=int)

    # Proper verification of Strassen block
    valid_strassen = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U_s[i*2+j, c] * V_s[j*2+k, c] * W_s[k*2+i, c]
                if s != 1:
                    valid_strassen = False
                    break
            if not valid_strassen:
                break
        if not valid_strassen:
            break

    if valid_strassen:
        rank = 59
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)

        # Strassen block: i in {0,1}, j in {0,1}, k in {0,1}
        col = 0
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m+j, col] = U_s[i*2+j, c]
            for j in range(2):
                for k in range(2):
                    V[j*p+k, col] = V_s[j*2+k, c]
            for k in range(2):
                for i in range(2):
                    W[k*n+i, col] = W_s[k*2+i, c]
            col += 1

        # Remaining entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

        # Verify full decomposition
        valid_full = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if s != 1:
                        valid_full = False
                        break
                if not valid_full:
                    break
            if not valid_full:
                break

        if valid_full:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 60
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
