import numpy as np

def construct():
    # Strassen 2x2 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1],
        [0, 1, 1, 0, 0, -1, 0],
        [-1, 0, 0, 0, -1, 0, 1]
    ])
    
    # Kronecker product to get 4x4 decomposition
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Convert to lists of integers
    U = U4.tolist()
    V = V4.tolist()
    W = W4.tolist()
    
    return {"U": U, "V": V, "W": W}
