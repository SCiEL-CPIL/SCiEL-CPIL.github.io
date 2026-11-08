import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U_S, V_S, W_S are 4x7
    U_S = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V_S = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ])
    
    W_S = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Schoolbook decomposition for 3x3x3
    # U_L, V_L, W_L are 9x27
    rank_L = 27
    U_L = np.zeros((9, rank_L))
    V_L = np.zeros((9, rank_L))
    W_L = np.zeros((9, rank_L))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_L[i * 3 + j, col] = 1
                V_L[j * 3 + k, col] = 1
                W_L[k * 3 + i, col] = 1
                col += 1
                
    # Construct 6x6x6 decomposition
    # Rank is 7 * 27 = 189
    rank_total = 7 * 27
    U = np.zeros((36, rank_total))
    V = np.zeros((36, rank_total))
    W = np.zeros((36, rank_total))
    
    for c_S in range(7):
        for c_L in range(27):
            c = c_S * 27 + c_L
            U[:, c] = np.kron(U_S[:, c_S], U_L[:, c_L])
            V[:, c] = np.kron(V_S[:, c_S], V_L[:, c_L])
            W[:, c] = np.kron(W_S[:, c_S], W_L[:, c_L])
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
