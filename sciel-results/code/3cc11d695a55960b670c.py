import numpy as np

def construct():
    n, m, p = 2, 2, 6
    rank = 21
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2
    # s: 0..6
    # alpha[s]: dict (i, j) -> val
    # beta[s]: dict (j, k) -> val
    # gamma[s]: dict (k, i) -> val
    
    alpha = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    beta = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    gamma = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1, (1,1): 1}
    ]
    
    for b in range(3):
        for s in range(7):
            r = b * 7 + s
            
            # Fill U (same for all blocks)
            for (i, j), val in alpha[s].items():
                U[i * m + j, r] = val
                
            # Fill V (shifted by block b)
            for (j, k_local), val in beta[s].items():
                k_global = b * 2 + k_local
                V[j * p + k_global, r] = val
                
            # Fill W (shifted by block b)
            for (k_local, i), val in gamma[s].items():
                k_global = b * 2 + k_local
                W[k_global * n + i, r] = val
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
