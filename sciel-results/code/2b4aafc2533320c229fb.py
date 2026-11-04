"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 8 Strassen factors for 3x3x3
    # U, V, W are 9x8
    U_s = np.array([
        [1, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=float)
    
    # This is not a valid Strassen decomposition. 
    # I will use the Schoolbook for the first two blocks as well to ensure correctness, 
    # resulting in rank 9+9+6=24.
    
    U_school = np.eye(9)
    V_school = np.eye(9)
    W_school = np.eye(9)
    
    # Block 0: cols 0-2
    U0 = U_school
    V0 = np.zeros((9, 9))
    W0 = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = i*3 + j
                v_idx = j*3 + k
                w_idx = k*3 + i
                V0[v_idx, idx] = 1
                W0[w_idx, idx] = 1
                
    # Block 1: cols 3-5
    U1 = U_school
    V1 = np.zeros((9, 9))
    W1 = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = i*3 + j
                v_idx = j*3 + (k+3)
                w_idx = (k+3)*3 + i
                V1[v_idx, idx] = 1
                W1[w_idx, idx] = 1
                
    # Block 2: cols 6-7
    U2 = np.zeros((9, 6))
    V2 = np.zeros((9, 6))
    W2 = np.zeros((6, 6)) # W is 6x6? No, W is (p*n, rank). 
    # For block 2, p=2, n=3. W size is 6x6.
    # But global W is 24x24.
    
    # Let's assemble the global matrices.
    # Rank = 9 + 9 + 6 = 24
    rank = 24
    U = np.zeros((9, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    # Block 0
    U[:, 0:9] = U0
    V[0:9, 0:9] = V0
    W[0:9, 0:9] = W0
    
    # Block 1
    U[:, 9:18] = U1
    V[3:12, 9:18] = V1 # V indices for j=0..2, k=3..5 are 3..11
    # Wait, V index is j*8 + k.
    # For block 1, k=3..5.
    # j=0, k=3..5 -> 3,4,5
    # j=1, k=3..5 -> 11,12,13
    # j=2, k=3..5 -> 19,20,21
    # My V1 construction above used j*3+k. This is wrong for global V.
    
    # Let's redo V and W indexing.
    
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    U = np.zeros((9, rank))
    
    # Block 0: k in 0..2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = i*3 + j
                u_idx = i*3 + j
                v_idx = j*8 + k
                w_idx = k*3 + i
                U[u_idx, c] = 1
                V[v_idx, c] = 1
                W[w_idx, c] = 1
                
    # Block 1: k in 3..5
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = 9 + i*3 + j
                u_idx = i*3 + j
                v_idx = j*8 + (k+3)
                w_idx = (k+3)*3 + i
                U[u_idx, c] = 1
                V[v_idx, c] = 1
                W[w_idx, c] = 1
                
    # Block 2: k in 6..7
    for i in range(3):
        for j in range(3):
            for k in range(2):
                c = 18 + i*3 + j*2 + k
                u_idx = i*3 + j
                v_idx = j*8 + (k+6)
                w_idx = (k+6)*3 + i
                U[u_idx, c] = 1
                V[v_idx, c] = 1
                W[w_idx, c] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
