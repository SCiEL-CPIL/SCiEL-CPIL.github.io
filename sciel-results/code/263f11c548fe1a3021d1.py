import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 73 decomposition using block structure:
    # Split A (3x5) into A_L (3x3) and A_R (3x2)
    # Split B (5x5) into B_LL (3x3), B_LR (3x2), B_RL (2x3), B_RR (2x2)
    # C = A_L B_LL + A_L B_LR + A_R B_RL + A_R B_RR
    # Ranks: 25 + 18 + 18 + 12 = 73
    
    rank = 73
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    
    # Block 1: A_L (3x3) * B_LL (3x3) -> C_L part (3x3)
    # i in 0..2, j in 0..2, k in 0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 5 + j, r] = 1
                V[j * 5 + k, r] = 1
                W[k * 3 + i, r] = 1
                r += 1
                
    # Block 2: A_L (3x3) * B_LR (3x2) -> C_R part (3x2)
    # i in 0..2, j in 0..2, k in 3..4
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                U[i * 5 + j, r] = 1
                V[j * 5 + k, r] = 1
                W[k * 3 + i, r] = 1
                r += 1
                
    # Block 3: A_R (3x2) * B_RL (2x3) -> C_L part (3x3)
    # i in 0..2, j in 3..4, k in 0..2
    for i in range(3):
        for j in range(3, 5):
            for k in range(3):
                U[i * 5 + j, r] = 1
                V[j * 5 + k, r] = 1
                W[k * 3 + i, r] = 1
                r += 1
                
    # Block 4: A_R (3x2) * B_RR (2x2) -> C_R part (3x2)
    # i in 0..2, j in 3..4, k in 3..4
    for i in range(3):
        for j in range(3, 5):
            for k in range(3, 5):
                U[i * 5 + j, r] = 1
                V[j * 5 + k, r] = 1
                W[k * 3 + i, r] = 1
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
