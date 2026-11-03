import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    # Strassen decomposition for 2x2x2: rank 7
    # U[i*2+j][c], V[j*2+k][c], W[k*2+i][c]
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],   # i=0,j=0
        [0, 0, 0, 0, 1, 0, 1],    # i=0,j=1
        [0, 1, 0, 0, 0, 1, 0],    # i=1,j=0
        [1, 1, 0, 1, 0, 0, -1]    # i=1,j=1
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],   # j=0,k=0
        [0, 0, 0, 1, 0, 1, 0],    # j=0,k=1
        [0, 0, 1, 0, 0, 0, 1],    # j=1,k=0
        [1, 0, -1, 0, 1, 0, 1]    # j=1,k=1
    ]
    W2 = [
        [1, 0, 0, 0, 0, 0, 0],    # k=0,i=0
        [0, 0, 0, 0, 1, 0, 0],    # k=0,i=1
        [0, 1, 1, 0, 0, 1, 0],    # k=1,i=0
        [1, 0, 0, 0, 0, 0, 1]     # k=1,i=1
    ]
    
    # Verify Strassen decomposition for 2x2
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook rank-64
        n, m, p = 4, 4, 4
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Construct rank-49 decomposition for 4x4x4 via Kronecker product of two Strassen
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * 4 + j][c] = U2[i1 * 2 + j1][c1] * U2[i2 * 2 + j2][c2]
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    V[j * 4 + k][c] = V2[j1 * 2 + k1][c1] * V2[j2 * 2 + k2][c2]
                                    W[k * 4 + i][c] = W2[k1 * 2 + i1][c1] * W2[k2 * 2 + i2][c2]
    
    # Verify the 4x4x4 decomposition
    valid4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(49):
                    s += U[i*4+j][c] * V[j*4+k][c] * W[k*4+i][c]
                if s != 1:
                    valid4 = False
                    break
            if not valid4:
                break
        if not valid4:
            break
    
    if not valid4:
        # Fallback to schoolbook rank-64
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
