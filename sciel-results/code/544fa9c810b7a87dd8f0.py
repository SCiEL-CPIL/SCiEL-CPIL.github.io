import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2
    # U2: 4x7, rows are A00, A01, A10, A11
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ], dtype=int)
    
    # V2: 4x7, rows are B00, B01, B10, B11
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ], dtype=int)
    
    # W2: 4x7, rows are C00, C10, C01, C11 (index k*n+i, n=2)
    # C00: idx 0
    # C10: idx 1
    # C01: idx 2
    # C11: idx 3
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ], dtype=int)
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    col = 0
    for r in range(7):
        for rp in range(7):
            U[:, col] = np.kron(U2[:, r], U2[:, rp])
            V[:, col] = np.kron(V2[:, r], V2[:, rp])
            W[:, col] = np.kron(W2[:, r], W2[:, rp])
            col += 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
