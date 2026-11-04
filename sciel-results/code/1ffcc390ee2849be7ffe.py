import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen 2x2x2 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Kronecker product for 4x4x4
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    n, m, p = DIMENSIONS
    rank = 98
    
    # Initialize full matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First block: k in [0, 4)
    # U is same for both blocks, so we just copy U4 to both halves
    U[:, :49] = U4
    U[:, 49:] = U4
    
    # V mapping:
    # V4 rows: j*4 + k (j in 0..3, k in 0..3)
    # V rows: j*8 + k (j in 0..3, k in 0..7)
    for j in range(4):
        for k in range(4):
            # First block
            V[j * 8 + k, :49] = V4[j * 4 + k, :]
            # Second block
            V[j * 8 + (k + 4), 49:] = V4[j * 4 + k, :]
            
    # W mapping:
    # W4 rows: k*4 + i (k in 0..3, i in 0..3)
    # W rows: k*4 + i (k in 0..7, i in 0..3)
    for k in range(4):
        for i in range(4):
            # First block
            W[k * 4 + i, :49] = W4[k * 4 + i, :]
            # Second block
            W[(k + 4) * 4 + i, 49:] = W4[k * 4 + i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
