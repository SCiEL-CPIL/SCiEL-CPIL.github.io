"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Strategy: Block decomposition
    # 15 columns = 7 blocks of size 2 (rank 7 each) + 1 block of size 1 (rank 4)
    # Total rank = 7 * 7 + 4 = 53
    
    # Pre-compute Strassen matrices for 2x2x2
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0], 
        [0, 0, 0, 0, 1, 0, 1],  
        [0, 1, 0, 0, 0, 1, 0],  
        [1, 1, 0, 1, 0, 0, -1]  
    ])
    
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0], 
        [0, 0, 1, 0, 0, 1, 0],  
        [0, 0, 0, 1, 0, 0, 1],  
        [1, 0, -1, 0, 1, 0, 1]  
    ])
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1], 
        [0, 1, 0, 1, 0, 0, 0],  
        [0, 0, 1, 0, 1, 0, 0],  
        [1, -1, 1, 0, 0, 1, 0]  
    ])
    
    # Pre-compute Schoolbook matrices for 2x2x1
    U_school = np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, 0], 
        [0, 0, 1, 0], 
        [0, 0, 0, 1]  
    ])
    
    V_school = np.array([
        [1, 0, 1, 0], 
        [0, 1, 0, 1]  
    ])
    
    W_school = np.array([
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ])
    
    total_rank = 7 * 7 + 4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    # 7 blocks of size 2
    for b in range(7):
        if time.time() > deadline:
            break
        k_start = 2 * b
        U[:, col_offset:col_offset+7] = U_strassen
        
        for j in range(m):
            for k_local in range(2):
                k_global = k_start + k_local
                V[j * p + k_global, col_offset:col_offset+7] = V_strassen[j * 2 + k_local, :]
        
        for k_local in range(2):
            k_global = k_start + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+7] = W_strassen[k_local * 2 + i, :]
        
        col_offset += 7
        
    # 1 block of size 1 at k=14
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    k_start = 14
    U[:, col_offset:col_offset+4] = U_school
    
    for j in range(m):
        V[j * p + k_start, col_offset:col_offset+4] = V_school[j, :]
        
    for i in range(n):
        W[k_start * n + i, col_offset:col_offset+4] = W_school[i, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
