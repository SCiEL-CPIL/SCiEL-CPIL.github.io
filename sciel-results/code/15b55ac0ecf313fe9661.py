"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Define the 2x2x2 Strassen tensor
    # U, V, W are 4x7 matrices
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, -1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Verify Strassen 2x2x2
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # U index: i*2+j
    # V index: j*2+k
    # W index: k*2+i
    
    # Construct 4x4x4 tensor using Kronecker product
    U4 = np.kron(U2, U2) # Shape 16x49
    V4 = np.kron(V2, V2) # Shape 16x49
    W4 = np.kron(W2, W2) # Shape 16x49
    
    # Construct 4x4x3 tensor using schoolbook
    # Rank 48
    U3 = np.zeros((16, 48), dtype=int)
    V3 = np.zeros((12, 48), dtype=int) # 4*3
    W3 = np.zeros((12, 48), dtype=int) # 3*4
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U3[i * 4 + j, col] = 1
                V3[j * 3 + k, col] = 1
                W3[k * 4 + i, col] = 1
                col += 1
                
    # Combine
    # Total rank 49 + 48 = 97
    total_rank = 97
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((28, total_rank), dtype=int) # 4*7
    W = np.zeros((28, total_rank), dtype=int) # 7*4
    
    # Part 1: 4x4x4 (k=0..3)
    U[:, :49] = U4
    # V4 is 16x49, rows j*4+k
    # V is 28x97, rows j*7+k
    for j in range(4):
        for k in range(4):
            V[j * 7 + k, :49] = V4[j * 4 + k, :]
            
    # W4 is 16x49, rows k*4+i
    # W is 28x97, rows k*4+i
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :49] = W4[k * 4 + i, :]
            
    # Part 2: 4x4x3 (k=4..6)
    U[:, 49:] = U3
    # V3 is 12x48, rows j*3+k'
    # V is 28x97, rows j*7+(4+k')
    for j in range(4):
        for kp in range(3):
            k = 4 + kp
            V[j * 7 + k, 49:] = V3[j * 3 + kp, :]
            
    # W3 is 12x48, rows k'*4+i
    # W is 28x97, rows (4+k')*4+i
    for kp in range(3):
        k = 4 + kp
        for i in range(4):
            W[k * 4 + i, 49:] = W3[kp * 4 + i, :]
            
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
