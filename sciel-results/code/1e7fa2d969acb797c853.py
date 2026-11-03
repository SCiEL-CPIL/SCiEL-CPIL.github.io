"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 factors
    # U: 4x7, V: 4x7, W: 4x7
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, -1]
    ])
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Schoolbook 3x3 factors
    # U: 9x27, V: 9x27, W: 9x27
    U_b = np.zeros((9, 27), dtype=int)
    V_b = np.zeros((9, 27), dtype=int)
    W_b = np.zeros((9, 27), dtype=int)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                col = i * 9 + j * 3 + k
                U_b[i * 3 + j, col] = 1
                V_b[j * 3 + k, col] = 1
                W_b[k * 3 + i, col] = 1
                
    # Kronecker product
    # U_total shape: (4*9, 7*27) = (36, 189)
    U_total = np.kron(U_s, U_b)
    V_total = np.kron(V_s, V_b)
    W_total = np.kron(W_s, W_b)
    
    return {"U": U_total, "V": V_total, "W": W_total}
# EVOLVE-BLOCK-END
