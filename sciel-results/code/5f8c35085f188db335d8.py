"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Factors for M_{1, 2, 4} (Rank 8)
    # A is 1x2, B is 2x4, C is 1x4
    # U1: 2x8, V1: 8x8, W1: 4x8
    U1 = np.zeros((2, 8))
    V1 = np.zeros((8, 8))
    W1 = np.zeros((4, 8))
    
    col = 0
    for j in range(2):
        for k in range(4):
            U1[j, col] = 1
            V1[j * 4 + k, col] = 1
            W1[k, col] = 1
            col += 1
            
    # Factors for M_{2, 2, 2} (Rank 7, Strassen)
    # A' is 2x2, B' is 2x2, C' is 2x2
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, -1],
        [1, 1, 0, 1, 0, 1, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Kronecker Products
    U_kron = np.kron(U1, U2) # Shape (2*4, 8*7) = (8, 56)
    V_kron = np.kron(V1, V2) # Shape (8*4, 8*7) = (32, 56)
    W_kron = np.kron(W1, W2) # Shape (4*4, 8*7) = (16, 56)
    
    # Permute rows to match standard index ordering
    # U: r_S = i2 * 4 + j1 * 2 + j2
    # r_K = j1 * 4 + i2 * 2 + j2
    # Map r_K to r_S
    perm_U = np.zeros(8, dtype=int)
    for j1 in range(2):
        for i2 in range(2):
            for j2 in range(2):
                r_K = j1 * 4 + i2 * 2 + j2
                r_S = i2 * 4 + j1 * 2 + j2
                perm_U[r_S] = r_K
    U_final = U_kron[perm_U, :]
    
    # V: s_S = j1 * 16 + k2 * 8 + k1 * 2 + l2
    # s_K = j1 * 16 + k1 * 4 + k2 * 2 + l2
    perm_V = np.zeros(32, dtype=int)
    for j1 in range(2):
        for k2 in range(2):
            for k1 in range(4):
                for l2 in range(2):
                    s_K = j1 * 16 + k1 * 4 + k2 * 2 + l2
                    s_S = j1 * 16 + k2 * 8 + k1 * 2 + l2
                    perm_V[s_S] = s_K
    V_final = V_kron[perm_V, :]
    
    # W: t_S = k1 * 4 + j2 * 2 + i2
    # t_K = k1 * 4 + i2 * 2 + j2
    perm_W = np.zeros(16, dtype=int)
    for k1 in range(4):
        for j2 in range(2):
            for i2 in range(2):
                t_K = k1 * 4 + i2 * 2 + j2
                t_S = k1 * 4 + j2 * 2 + i2
                perm_W[t_S] = t_K
    W_final = W_kron[perm_W, :]
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
