import numpy as np

def construct():
    n, m, p = 3, 3, 4
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    R = 35
    
    U = np.zeros((dim_u, R))
    V = np.zeros((dim_v, R))
    W = np.zeros((dim_w, R))
    
    # Strassen terms for i in {0,1}, j in {0,1}, k in {0,1}
    # M1
    U[0, 0] += 1; U[4, 0] += 1
    V[0, 0] += 1; V[5, 0] += 1
    W[0, 0] += 1; W[4, 0] += 1
    
    # M2
    U[3, 1] += 1; U[4, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1; W[4, 1] += -1
    
    # M3
    U[0, 2] += 1
    V[1, 2] += 1; V[5, 2] += -1
    W[3, 2] += 1
    
    # M4
    U[4, 3] += 1
    V[4, 3] += 1; V[0, 3] += -1
    W[0, 3] += 1; W[1, 3] += 1
    
    # M5
    U[0, 4] += 1; U[1, 4] += 1
    V[5, 4] += 1
    W[0, 4] += -1; W[3, 4] += 1
    
    # M6
    U[3, 5] += 1; U[0, 5] += -1
    V[0, 5] += 1; V[1, 5] += 1
    W[4, 5] += 1
    
    # M7
    U[1, 6] += 1; U[4, 6] += -1
    V[4, 6] += 1; V[5, 6] += 1
    W[0, 6] += 1; W[3, 6] += 1
    
    # Other terms
    col = 7
    # Group 2: i in {0,1}, j=2, k in {0,1}
    for i in [0, 1]:
        for k in [0, 1]:
            u = i * m + 2
            v = 2 * p + k
            w = k * n + i
            U[u, col] = 1
            V[v, col] = 1
            W[w, col] = 1
            col += 1
            
    # Group 3: i=2, all j, k
    for j in range(m):
        for k in range(p):
            u = 2 * m + j
            v = j * p + k
            w = k * n + 2
            U[u, col] = 1
            V[v, col] = 1
            W[w, col] = 1
            col += 1
            
    # Group 4: i in {0,1}, k in {2,3}, all j
    for i in [0, 1]:
        for k in [2, 3]:
            for j in range(m):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1
                V[v, col] = 1
                W[w, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
