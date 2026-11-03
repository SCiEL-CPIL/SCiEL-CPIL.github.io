import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # Strassen 2x2x2 decomposition
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Kronecker product for 4x4x4
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Now build U, V, W for 4x4x9
    # Rank = 49 (for B1) + 49 (for B2) + 16 (for B3)
    rank = 49 + 49 + 16
    
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # Part 1: A * B1 (cols 0-3)
    # U is same as U4
    U[:, 0:49] = U4
    
    # V: Map B1 (indices in 4x4) to B (indices in 4x9)
    # B1[i,j] is at B[i, j] -> index i*9 + j
    # U4/V4/W4 use index i*4 + j
    for i in range(4):
        for j in range(4):
            idx4 = i * 4 + j
            idx9 = i * 9 + j
            V[idx9, 0:49] = V4[idx4, :]
            W[idx9, 0:49] = W4[idx4, :]
            
    # Part 2: A * B2 (cols 4-7)
    # U is same as U4
    U[:, 49:98] = U4
    
    # V: Map B2 (indices in 4x4) to B (indices in 4x9)
    # B2[i,j] is at B[i, j+4] -> index i*9 + (j+4)
    for i in range(4):
        for j in range(4):
            idx4 = i * 4 + j
            idx9 = i * 9 + (j + 4)
            V[idx9, 49:98] = V4[idx4, :]
            W[idx9, 49:98] = W4[idx4, :]
            
    # Part 3: A * B3 (col 8)
    # Schoolbook
    for i in range(4):
        for j in range(4):
            col = 98 + (i * 4 + j)
            U[i * 4 + j, col] = 1
            V[j * 9 + 8, col] = 1
            W[i * 9 + 8, col] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
