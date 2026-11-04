import numpy as np

def construct():
    # Strassen 2x2x2 decomposition
    # U: 4x7, V: 4x7, W: 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0]
    ], dtype=int)
    
    # Kronecker product for 4x4x4
    # U4: 16x49
    U4 = np.kron(U2, U2)
    # V4: 16x49
    V4 = np.kron(V2, V2)
    # W4: 16x49
    W4 = np.kron(W2, W2)
    
    # Construct 4x4x8 decomposition
    # Rank 98
    # U: 16x98
    U = np.zeros((16, 98), dtype=int)
    U[:, 0:49] = U4
    U[:, 49:98] = U4
    
    # V: 32x98
    V = np.zeros((32, 98), dtype=int)
    for j in range(4):
        for k in range(4):
            v_idx = j * 4 + k
            # First block of columns (0-3)
            v_idx_8_first = j * 8 + k
            V[v_idx_8_first, 0:49] = V4[v_idx, :]
            # Second block of columns (4-7)
            v_idx_8_second = j * 8 + k + 4
            V[v_idx_8_second, 49:98] = V4[v_idx, :]
            
    # W: 32x98
    W = np.zeros((32, 98), dtype=int)
    # First block of output columns (0-3)
    W[0:16, 0:49] = W4
    # Second block of output columns (4-7)
    W[16:32, 49:98] = W4
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
