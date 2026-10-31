import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define the 7 Strassen factors for 2x2 matrix multiplication
    # Indices:
    # U (A): i*2 + j
    # V (B): j*2 + k
    # W (C): k*2 + i
    
    # U2[r] is 2x2 matrix for factor r
    U2 = [
        np.array([[1, 0], [0, 0]], dtype=int), # 1: A00
        np.array([[1, 0], [1, 0]], dtype=int), # 2: A00 + A10
        np.array([[0, 1], [0, 1]], dtype=int), # 3: A01 + A11
        np.array([[0, 0], [0, 1]], dtype=int), # 4: A11
        np.array([[1, 0], [0, 1]], dtype=int), # 5: A00 + A11
        np.array([[0, 1], [0, -1]], dtype=int),# 6: A01 - A11
        np.array([[1, 0], [-1, 0]], dtype=int) # 7: A00 - A10
    ]
    
    V2 = [
        np.array([[1, 0], [0, 1]], dtype=int), # 1: B00 + B11
        np.array([[0, 0], [0, 1]], dtype=int), # 2: B11
        np.array([[1, 0], [0, 0]], dtype=int), # 3: B00
        np.array([[0, 1], [1, 0]], dtype=int), # 4: B01 + B10
        np.array([[1, 0], [0, 1]], dtype=int), # 5: B00 + B11
        np.array([[0, 0], [1, 1]], dtype=int), # 6: B10 + B11
        np.array([[1, 1], [0, 0]], dtype=int)  # 7: B00 + B01
    ]
    
    W2 = [
        np.array([[1, 0], [0, 1]], dtype=int), # 1: C00 + C11 (coeff 1, 1) -> C00=1, C11=1
        np.array([[0, 0], [1, 0]], dtype=int), # 2: C10 (coeff 1) -> C10=1
        np.array([[0, 1], [0, 1]], dtype=int), # 3: C01 + C11 (coeff 1, 1) -> C01=1, C11=1
        np.array([[1, 0], [1, 0]], dtype=int), # 4: C00 + C10 (coeff 1, 1) -> C00=1, C10=1
        np.array([[-1, 1], [0, 0]], dtype=int),# 5: -C00 + C01 (coeff -1, 1) -> C00=-1, C01=1
        np.array([[0, 0], [0, 1]], dtype=int), # 6: C11 (coeff 1) -> C11=1
        np.array([[-1, 0], [0, 0]], dtype=int) # 7: -C00 (coeff -1) -> C00=-1
    ]
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    col = 0
    for r in range(7):
        for s in range(7):
            # Kronecker product of the 2x2 matrices
            # This constructs the 4x4 factor for the 4x4 multiplication
            u_col = np.kron(U2[r], U2[s]).flatten()
            v_col = np.kron(V2[r], V2[s]).flatten()
            w_col = np.kron(W2[r], W2[s]).flatten()
            
            U[:, col] = u_col
            V[:, col] = v_col
            W[:, col] = w_col
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
