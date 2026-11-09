import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # Define 2x2 Strassen factors
    # U2[i*2+j, col], V2[j*2+k, col], W2[k*2+i, col]
    
    U2 = [
        [1, 1, -1, 1, 0, 1, 0], # i=0, j=0
        [0, 0, 0, 0, 0, 1, 0], # i=0, j=1
        [0, 0, 0, 0, 0, 0, 1], # i=1, j=0
        [0, 1, 1, 0, 1, 0, 1]  # i=1, j=1
    ]
    
    V2 = [
        [1, 1, -1, -1, 1, 0, 1], # j=0, k=0
        [0, 0, 0, 0, 0, 0, 0],   # j=0, k=1
        [0, 0, 0, 0, 0, 0, -1],  # j=1, k=0
        [0, 1, 1, 1, -1, 1, 0]   # j=1, k=1
    ]
    
    W2 = [
        [1, -1, 1, 0, 1, 0, 0],  # k=0, i=0
        [0, 0, 0, 1, 0, 1, 0],   # k=0, i=1
        [0, 0, 0, 0, 1, 0, 1],   # k=1, i=0
        [0, 1, -1, 1, 0, 0, -1]  # k=1, i=1
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct 4x4 factors
    # U4[i*4+j, c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    # i = 2*i1 + i2, j = 2*j1 + j2
    # c = 7*c1 + c2
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2 * i1 + i2
                    j = 2 * j1 + j2
                    idx_U = i * m + j
                    
                    for c1 in range(7):
                        for c2 in range(7):
                            c = 7 * c1 + c2
                            val = U2[i1 * 2 + j1][c1] * U2[i2 * 2 + j2][c2]
                            U[idx_U][c] = val
                            
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2 * j1 + j2
                    k = 2 * k1 + k2
                    idx_V = j * p + k
                    
                    for c1 in range(7):
                        for c2 in range(7):
                            c = 7 * c1 + c2
                            val = V2[j1 * 2 + k1][c1] * V2[j2 * 2 + k2][c2]
                            V[idx_V][c] = val
                            
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2 * k1 + k2
                    i = 2 * i1 + i2
                    idx_W = k * n + i
                    
                    for c1 in range(7):
                        for c2 in range(7):
                            c = 7 * c1 + c2
                            val = W2[k1 * 2 + i1][c1] * W2[k2 * 2 + i2][c2]
                            W[idx_W][c] = val
                            
    return {"U": U, "V": V, "W": W}
