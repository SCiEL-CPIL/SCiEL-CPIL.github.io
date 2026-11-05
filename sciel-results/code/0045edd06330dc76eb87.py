import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    # U_strassen: 4x7, V_strassen: 4x7, W_strassen: 4x7
    # Indices: U[i*2+j, col], V[j*2+k, col], W[k*2+i, col]
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1 # C11
    W_s[3, 0] = 1 # C22
    
    # M2 = (A21+A22)B11
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1 # C21
    W_s[3, 1] = -1 # C22
    
    # M3 = A11(B12-B22)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1 # C12
    W_s[3, 2] = 1 # C22
    
    # M4 = A22(B21-B11)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1 # C11
    W_s[2, 3] = 1 # C21
    
    # M5 = (A11+A12)B22
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1 # C11
    W_s[1, 4] = 1 # C12
    
    # M6 = (A21-A11)(B11+B12)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1 # C22
    
    # M7 = (A12-A22)(B21+B22)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1 # C11
    W_s[1, 6] = -1 # C12
    
    # Construct 4x4x4 decomposition using Kronecker product
    # Rank 49
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            U_444[:, c] = np.kron(U_s[:, c1], U_s[:, c2])
            V_444[:, c] = np.kron(V_s[:, c1], V_s[:, c2])
            W_444[:, c] = np.kron(W_s[:, c1], W_s[:, c2])
            
    # Total rank 85
    rank = 85
    U = np.zeros((20, rank))
    V = np.zeros((25, rank))
    W = np.zeros((20, rank))
    
    # Part 1: A1 B11 (4x4x4)
    # A1 is A[0:4, 0:4]. B11 is B[0:4, 0:4]. C adds to C[0:4, 0:4].
    # U indices: i*5+j. i in 0..3, j in 0..3.
    # V indices: j*5+k. j in 0..3, k in 0..3.
    # W indices: k*4+i. k in 0..3, i in 0..3.
    
    for c in range(49):
        col = c
        # Map U_444 indices (i*4+j) to global U (i*5+j)
        for local_i in range(4):
            for local_j in range(4):
                local_idx = local_i * 4 + local_j
                global_idx = local_i * 5 + local_j
                U[global_idx, col] = U_444[local_idx, c]
                
        # Map V_444 indices (j*4+k) to global V (j*5+k)
        for local_j in range(4):
            for local_k in range(4):
                local_idx = local_j * 4 + local_k
                global_idx = local_j * 5 + local_k
                V[global_idx, col] = V_444[local_idx, c]
                
        # Map W_444 indices (k*4+i) to global W (k*4+i)
        for local_k in range(4):
            for local_i in range(4):
                local_idx = local_k * 4 + local_i
                global_idx = local_k * 4 + local_i
                W[global_idx, col] = W_444[local_idx, c]
                
    # Part 2: A1 B12 (4x4x1)
    # A1 is A[0:4, 0:4]. B12 is B[0:4, 4]. C adds to C[0:4, 4].
    # Rank 16
    col_offset = 49
    count = 0
    for i in range(4):
        for j in range(4):
            col = col_offset + count
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1 # k=4, i=i
            count += 1
            
    # Part 3: A2 B21 (4x1x4)
    # A2 is A[0:4, 4]. B21 is B[4, 0:4]. C adds to C[0:4, 0:4].
    # Rank 16
    col_offset = 49 + 16
    count = 0
    for i in range(4):
        for k in range(4):
            col = col_offset + count
            U[i * 5 + 4, col] = 1
            V[4 * 5 + k, col] = 1
            W[k * 4 + i, col] = 1
            count += 1
            
    # Part 4: A2 B22 (4x1x1)
    # A2 is A[0:4, 4]. B22 is B[4, 4]. C adds to C[0:4, 4].
    # Rank 4
    col_offset = 49 + 16 + 16
    count = 0
    for i in range(4):
        col = col_offset + count
        U[i * 5 + 4, col] = 1
        V[4 * 5 + 4, col] = 1
        W[4 * 4 + i, col] = 1
        count += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
