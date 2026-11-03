import numpy as np

def construct():
    # 2x2 Strassen vectors
    U2 = [
        [1, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V2 = [
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W2 = [
        [1, 0, 0, 1],
        [0, 0, 0, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Convert to numpy arrays
    U2 = np.array(U2)
    V2 = np.array(V2)
    W2 = np.array(W2)
    
    # Kronecker product to get 4x4 decomposition
    # Rank = 7 * 7 = 49
    U4 = []
    V4 = []
    W4 = []
    
    for i in range(7):
        for j in range(7):
            U4.append(np.kron(U2[i], U2[j]))
            V4.append(np.kron(V2[i], V2[j]))
            W4.append(np.kron(W2[i], W2[j]))
            
    U4 = np.array(U4).T # Shape (16, 49)
    V4 = np.array(V4).T # Shape (16, 49)
    W4 = np.array(W4).T # Shape (16, 49)
    
    return {'U': U4, 'V': V4, 'W': W4}
