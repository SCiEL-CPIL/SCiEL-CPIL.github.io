import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 matrices
    # U, V, W are 4x7 matrices
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],  # C00 (k=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (k=0, i=1) -> index 1
        [0, 0, 1, 0, 1, 0, 0],   # C01 (k=1, i=0) -> index 2
        [1, 1, 1, 0, 0, 1, 0]    # C11 (k=1, i=1) -> index 3
    ]
    
    # Convert to numpy for efficient Kronecker product
    S_U_np = np.array(S_U)
    S_V_np = np.array(S_V)
    S_W_np = np.array(S_W)
    
    # Kronecker product to get 4x4 decomposition
    # U shape (16, 49), V shape (16, 49), W shape (16, 49)
    U = np.kron(S_U_np, S_U_np)
    V = np.kron(S_V_np, S_V_np)
    W = np.kron(S_W_np, S_W_np)
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
