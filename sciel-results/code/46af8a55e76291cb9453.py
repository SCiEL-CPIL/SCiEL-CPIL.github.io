import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    # Strassen decomposition for 2x2x2
    # U, V, W are 4x7 matrices
    U_222 = np.array([
        [1, 1, 1, 0, 0, 1, -1],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, -1, 1],
        [1, 1, 0, 1, 1, 0, 0]
    ], dtype=int)
    
    V_222 = np.array([
        [1, 0, 0, -1, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, -1],
        [1, 0, -1, 0, 0, 1, 0]
    ], dtype=int)
    
    W_222 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 0, -1, 0, 0, -1, 0]
    ], dtype=int)
    
    # Construct 2x2x4 decomposition
    # Rank 14
    # U_224: 4x14
    # V_224: 8x14
    # W_224: 8x14
    
    U_224 = np.hstack([U_222, U_222])
    
    V_224 = np.zeros((8, 14), dtype=int)
    V_224[:4, :7] = V_222
    V_224[4:, 7:] = V_222
    
    W_224 = np.zeros((8, 14), dtype=int)
    W_224[:4, :7] = W_222
    W_224[4:, 7:] = W_222
    
    # Tensor product to get 4x4x8
    # U_total = U_222 (outer) x U_224 (inner)
    # Dimensions: (4*4) x (7*14) = 16 x 98
    
    U_total = np.kron(U_222, U_224)
    V_total = np.kron(V_222, V_224)
    W_total = np.kron(W_222, W_224)
    
    # Convert to lists for the interface
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
