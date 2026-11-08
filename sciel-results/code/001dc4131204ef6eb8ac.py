import numpy as np

def construct():
    # Strassen decomposition for 2x2x2
    # U, V, W are 4x7 matrices
    # Indices: U[2*i+j, c], V[2*j+k, c], W[2*k+i, c]
    
    U_base = np.array([
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1
        [1, 1, 0, 0, 0, 1, 0],  # i=1, j=0
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1
    ])
    
    V_base = np.array([
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0
        [0, 0, 1, 0, 0, 1, 0],  # j=0, k=1
        [1, 0, -1, 1, 0, 0, 1], # j=1, k=0
        [1, 0, 0, 0, 1, 0, 1]   # j=1, k=1
    ])
    
    W_base = np.array([
        [1, 0, 0, 1, -1, 0, 1], # k=0, i=0
        [0, 0, 1, 0, 1, 0, 0],  # k=0, i=1
        [0, 1, 0, 1, 0, 0, 0],  # k=1, i=0
        [1, 0, 1, 0, 0, 1, 1]   # k=1, i=1
    ])
    
    # 2x4x4 problem
    # U: 8x28 (i in 0..1, j in 0..3)
    # V: 16x28 (j in 0..3, k in 0..3)
    # W: 8x28 (k in 0..3, i in 0..1)
    
    U = np.zeros((8, 28))
    V = np.zeros((16, 28))
    W = np.zeros((8, 28))
    
    # Block 1: A1 (j 0..1) * B11 (j 0..1, k 0..1) -> C (k 0..1)
    # U rows: 4i+j for j in 0..1
    # V rows: 4j+k for j in 0..1, k in 0..1
    # W rows: 2k+i for k in 0..1
    
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[4*i + j, c] = U_base[2*i + j, c]
        
        for j in range(2):
            for k in range(2):
                V[4*j + k, c] = V_base[2*j + k, c]
        
        for k in range(2):
            for i in range(2):
                W[2*k + i, c] = W_base[2*k + i, c]
                
    # Block 2: A1 (j 0..1) * B21 (j 2..3, k 0..1) -> C (k 0..1)
    # U rows: same as Block 1
    # V rows: 4j+k for j in 2..3, k in 0..1
    # W rows: same as Block 1
    
    c_off = 7
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[4*i + j, c + c_off] = U_base[2*i + j, c]
        
        for j in range(2):
            for k in range(2):
                V[4*(j+2) + k, c + c_off] = V_base[2*j + k, c]
        
        for k in range(2):
            for i in range(2):
                W[2*k + i, c + c_off] = W_base[2*k + i, c]

    # Block 3: A2 (j 2..3) * B12 (j 0..1, k 2..3) -> C (k 2..3)
    # U rows: 4i+j for j in 2..3
    # V rows: 4j+k for j in 0..1, k in 2..3
    # W rows: 2k+i for k in 2..3
    
    c_off = 14
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[4*i + (j+2), c + c_off] = U_base[2*i + j, c]
        
        for j in range(2):
            for k in range(2):
                V[4*j + (k+2), c + c_off] = V_base[2*j + k, c]
        
        for k in range(2):
            for i in range(2):
                W[2*(k+2) + i, c + c_off] = W_base[2*k + i, c]

    # Block 4: A2 (j 2..3) * B22 (j 2..3, k 2..3) -> C (k 2..3)
    # U rows: same as Block 3
    # V rows: 4j+k for j in 2..3, k in 2..3
    # W rows: same as Block 3
    
    c_off = 21
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[4*i + (j+2), c + c_off] = U_base[2*i + j, c]
        
        for j in range(2):
            for k in range(2):
                V[4*(j+2) + (k+2), c + c_off] = V_base[2*j + k, c]
        
        for k in range(2):
            for i in range(2):
                W[2*(k+2) + i, c + c_off] = W_base[2*k + i, c]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
