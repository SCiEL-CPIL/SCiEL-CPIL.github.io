"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24
    
    # 1. Define Strassen decomposition for 2x2x2 (Rank 7)
    # U2 (4x7), V2 (4x7), W2 (4x7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, -1, -1]  # A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ], dtype=int)
    
    # 2. Construct 4x4x4 decomposition (Rank 49) via Kronecker product
    # U4 = kron(U2, U2), etc.
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # 3. Construct 4x4x1 decomposition (Rank 16)
    # U_441 (16x16), V_441 (4x16), W_441 (4x16)
    U_441 = np.eye(16, dtype=int)
    V_441 = np.zeros((4, 16), dtype=int)
    W_441 = np.zeros((4, 16), dtype=int)
    for c in range(16):
        i = c // 4
        j = c % 4
        V_441[j, c] = 1
        W_441[i, c] = 1
        
    # 4. Assemble Part 1: 4x4x4 (Rank 49)
    # U1 (20x49), V1 (30x49), W1 (24x49)
    U1 = np.zeros((20, 49), dtype=int)
    V1 = np.zeros((30, 49), dtype=int)
    W1 = np.zeros((24, 49), dtype=int)
    
    for i in range(4):
        for j in range(4):
            # U index: i*5 + j
            # U4 index: i*4 + j
            U1[i*5 + j, :] = U4[i*4 + j, :]
            
            for k in range(4):
                # V index: j*6 + k
                # V4 index: j*4 + k
                V1[j*6 + k, :] = V4[j*4 + k, :]
                
                # W index: k*4 + i
                # W4 index: k*4 + i
                W1[k*4 + i, :] = W4[k*4 + i, :]

    # 5. Assemble Part 2: 4x4x2 (Rank 32)
    # This consists of two 4x4x1 blocks for k=4 and k=5.
    # U2 (20x32), V2 (30x32), W2 (24x32)
    U2 = np.zeros((20, 32), dtype=int)
    V2 = np.zeros((30, 32), dtype=int)
    W2 = np.zeros((24, 32), dtype=int)
    
    # Block k=4 (cols 0-15 of rank)
    for i in range(4):
        for j in range(4):
            c_local = i*4 + j
            c_global = c_local # 0-15
            U2[i*5 + j, c_global] = 1
            V2[j*6 + 4, c_global] = 1
            W2[4*4 + i, c_global] = 1 # k=4, i
            
    # Block k=5 (cols 16-31 of rank)
    for i in range(4):
        for j in range(4):
            c_local = i*4 + j
            c_global = 16 + c_local
            U2[i*5 + j, c_global] = 1
            V2[j*6 + 5, c_global] = 1
            W2[5*4 + i, c_global] = 1 # k=5, i

    # 6. Assemble Part 3: 4x1x6 (Rank 1)
    # U3 (20x1), V3 (30x1), W3 (24x1)
    U3 = np.zeros((20, 1), dtype=int)
    V3 = np.zeros((30, 1), dtype=int)
    W3 = np.zeros((24, 1), dtype=int)
    
    for i in range(4):
        U3[i*5 + 4, 0] = 1 # j=4
    for k in range(6):
        V3[4*6 + k, 0] = 1 # j=4
    for i in range(4):
        for k in range(6):
            W3[k*4 + i, 0] = 1

    # 7. Combine all parts
    U_total = np.hstack([U1, U2, U3])
    V_total = np.hstack([V1, V2, V3])
    W_total = np.hstack([W1, W2, W3])
    
    # Verify time budget
    if time.time() - time.time() > 0: # Placeholder
        pass

    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
# EVOLVE-BLOCK-END
