import numpy as np

def construct():
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen factors for 2x2x2
    # u: 4x7, v: 4x7, w: 4x7
    u = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, -1, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, 0, 0]
    ])
    v = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    w = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0]
    ])
    
    # Fill block 1 (k=0,1)
    for r in range(7):
        U[:, r] = u[:, r]
        V[0:4, r] = v[:, r] # j=0,1, k=0,1
        V[4:8, r] = 0       # j=0,1, k=2,3
        W[0:4, r] = w[:, r] # k=0,1, i=0,1
        W[4:8, r] = 0       # k=2,3, i=0,1
        
    # Fill block 2 (k=2,3)
    for r in range(7):
        U[:, r + 7] = u[:, r]
        V[0:4, r + 7] = 0       # j=0,1, k=0,1
        V[4:8, r + 7] = v[:, r] # j=0,1, k=2,3
        W[0:4, r + 7] = 0       # k=0,1, i=0,1
        W[4:8, r + 7] = w[:, r] # k=2,3, i=0,1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
