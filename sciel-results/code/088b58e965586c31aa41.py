import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0

def construct():
    n, m, p = 2, 2, 8
    
    # Strassen factors for 2x2 matrix multiplication
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Rows are indices, Cols are rank terms
    
    U_s = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1],
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ], dtype=int)
    
    V_s = np.array([
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ], dtype=int)
    
    W_s = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, -1],
        [0, 1, 0, 1], 
        [1, 0, 1, 0], 
        [-1, 1, 0, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ], dtype=int)
    
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col_offset = 0
    for t in range(num_blocks):
        U[:, col_offset:col_offset+7] = U_s.T
        
        v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
        for r in range(7):
            for l in range(4):
                V[v_indices[l], col_offset + r] = V_s[r, l]
                
        w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
        for r in range(7):
            for l in range(4):
                W[w_indices[l], col_offset + r] = W_s[r, l]
                
        col_offset += 7
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
