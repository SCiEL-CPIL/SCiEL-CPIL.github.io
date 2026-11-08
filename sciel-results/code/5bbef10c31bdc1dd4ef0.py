import numpy as np

def construct():
    # Define 2x2 Strassen decomposition
    # U2, V2, W2 are 4x7 matrices
    # Indices for U2: row i*2+j
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    # Indices for V2: row j*2+k
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    # Indices for W2: row k*2+i
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 1]
    ]

    # Convert to numpy arrays for Kronecker product
    U2_np = np.array(U2)
    V2_np = np.array(V2)
    W2_np = np.array(W2)

    # Kronecker product
    U4 = np.kron(U2_np, U2_np)
    V4 = np.kron(V2_np, V2_np)
    W4 = np.kron(W2_np, W2_np)

    # Convert back to lists of lists of integers
    U = U4.tolist()
    V = V4.tolist()
    W = W4.tolist()

    return {"U": U, "V": V, "W": W}
