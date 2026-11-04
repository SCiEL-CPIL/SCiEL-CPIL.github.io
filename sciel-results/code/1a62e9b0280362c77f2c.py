import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Winograd 2x2 matrices
    # U: Input A transform (4x4)
    U2 = np.array([
        [1, 1, 1, 1],
        [1, -1, -1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, -1]
    ], dtype=int)
    
    # V: Input B transform (4x4)
    V2 = np.array([
        [1, 1, 1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1]
    ], dtype=int)
    
    # W: Output C transform (4x4)
    W2 = np.array([
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1]
    ], dtype=int)
    
    # Kronecker product 3 times to get 8x8 decomposition
    # U8 = U2 (x) U2 (x) U2
    U8 = np.kron(np.kron(U2, U2), U2)
    V8 = np.kron(np.kron(V2, V2), V2)
    W8 = np.kron(np.kron(W2, W2), W2)
    
    # Extract 6x6 sub-blocks
    # U8 has shape (64, 64). Rows are indexed by i*8 + j.
    # We need rows for i in 0..5, j in 0..5.
    U_rows = []
    for i in range(6):
        for j in range(6):
            row_idx = i * 8 + j
            U_rows.append(U8[row_idx])
    U = np.array(U_rows)
    
    V_rows = []
    for j in range(6):
        for k in range(6):
            row_idx = j * 8 + k
            V_rows.append(V8[row_idx])
    V = np.array(V_rows)
    
    W_rows = []
    for k in range(6):
        for i in range(6):
            row_idx = k * 8 + i
            W_rows.append(W8[row_idx])
    W = np.array(W_rows)
    
    # Convert to lists of lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
