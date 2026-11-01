import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize U, V, W
    # U: (n*m, rank) -> (4, 7)
    # V: (m*p, rank) -> (4, 7)
    # W: (p*n, rank) -> (4, 7)
    
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    factors = [
        # M1
        ([ (0,0), (1,1) ], [1, 1], 
         [ (0,0), (1,1) ], [1, 1], 
         [ (0,0), (1,1) ], [1, 1]),
        # M2
        ([ (1,0), (1,1) ], [1, 1], 
         [ (0,0) ], [1], 
         [ (0,1), (1,1) ], [1, -1]),
        # M3
        ([ (0,0) ], [1], 
         [ (0,1), (1,1) ], [1, -1], 
         [ (1,0), (1,1) ], [1, 1]),
        # M4
        ([ (1,1) ], [1], 
         [ (1,0), (0,0) ], [1, -1], 
         [ (0,0), (0,1) ], [1, 1]),
        # M5
        ([ (0,0), (0,1) ], [1, 1], 
         [ (1,1) ], [1], 
         [ (0,0), (1,0) ], [-1, 1]),
        # M6
        ([ (1,0), (0,0) ], [1, -1], 
         [ (0,0), (0,1) ], [1, 1], 
         [ (1,1) ], [1]),
        # M7
        ([ (0,1), (1,1) ], [1, -1], 
         [ (1,0), (1,1) ], [1, 1], 
         [ (0,0) ], [1]),
    ]
    
    for r, (u_c, u_v, v_c, v_v, w_c, w_v) in enumerate(factors):
        for (i, j), val in zip(u_c, u_v):
            U[i*m + j, r] = val
        for (j, k), val in zip(v_c, v_v):
            V[j*p + k, r] = val
        for (k, i), val in zip(w_c, w_v):
            W[k*n + i, r] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
