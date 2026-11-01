import numpy as np

def construct():
    n, m, p = 2, 2, 10
    num_blocks = p // 2
    
    # Strassen terms
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    rank = num_blocks * 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(num_blocks):
        for t in range(7):
            col = r * 7 + t
            u_entries, v_entries, w_entries = strassen_terms[t]
            
            for (i, j), val in u_entries.items():
                U[i * m + j, col] = val
            
            for (j, k), val in v_entries.items():
                V[j * p + (2 * r + k), col] = val
            
            for (k, i), val in w_entries.items():
                W[(2 * r + k) * n + i, col] = val
                
    return {"U": U, "V": V, "W": W}
