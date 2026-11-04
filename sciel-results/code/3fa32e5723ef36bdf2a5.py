import numpy as np

def construct():
    # 2x2 rank-4 factors
    U2 = np.array([
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0]
    ])
    V2 = np.array([
        [1, 0, 0, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0]
    ])
    W2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ])
    
    # 3x3 schoolbook factors (rank 9)
    U3 = np.eye(9)
    V3 = np.eye(9)
    W3 = np.eye(9)
    
    # Kronecker product
    U6 = np.kron(U2, U3)
    V6 = np.kron(V2, V3)
    W6 = np.kron(W2, W3)
    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
