import numpy as np

def construct():
    n, m, p = 5, 5, 5
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Rank calculation:
    # 3x3x3 schoolbook: 3*3*3 = 27
    # i<3, k in {3,4}: 3*2*5 = 30
    # i in {3,4}, k<3: 2*3*5 = 30
    # i in {3,4}, k in {3,4}: 2*2*5 = 20
    # Total: 27 + 30 + 30 + 20 = 107
    
    R = 107
    U = np.zeros((size_u, R))
    V = np.zeros((size_v, R))
    W = np.zeros((size_w, R))
    
    r = 0
    
    # 1. 3x3x3 block (i, j, k in 0..2)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
                r += 1
                
    # 2. i < 3, k in {3, 4}
    for i in range(3):
        for k in range(3, 5):
            for j in range(5):
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
                r += 1
                
    # 3. i in {3, 4}, k < 3
    for i in range(3, 5):
        for k in range(3):
            for j in range(5):
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
                r += 1
                
    # 4. i in {3, 4}, k in {3, 4}
    for i in range(3, 5):
        for k in range(3, 5):
            for j in range(5):
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
