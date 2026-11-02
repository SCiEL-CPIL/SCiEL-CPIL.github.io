import numpy as np

def construct():
    n, m, p = 4, 4, 5
    
    # Strassen 2x2x2 decomposition
    # U, V, W are 4x7
    U22 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V22 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    W22 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Kronecker product for 4x4x4
    U44 = np.kron(U22, U22)
    V44 = np.kron(V22, V22)
    W44 = np.kron(W22, W22)
    
    # Schoolbook for 4x4x1 (5th column)
    # U441: 16x16, V441: 4x16, W441: 4x16
    U441 = np.eye(16)
    V441 = np.zeros((4, 16))
    W441 = np.zeros((4, 16))
    
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            V441[j, idx] = 1
            W441[i, idx] = 1
            
    # Assemble final matrices
    # U: 16 x 65
    U = np.zeros((16, 65))
    U[:, :49] = U44
    U[:, 49:] = U441
    
    # V: 20 x 65
    V = np.zeros((20, 65))
    V[:16, :49] = V44
    V[16:20, 49:] = V441
    
    # W: 20 x 65
    W = np.zeros((20, 65))
    W[:16, :49] = W44
    W[16:20, 49:] = W441
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
