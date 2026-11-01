import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's algorithm decomposition for 2x2 matrix multiplication
    # Rank 7 is the theoretical minimum for 2x2 matrix multiplication.
    # We return the standard Strassen decomposition.
    
    n, m, p = DIMENSIONS
    rank = 7
    
    # U shape: (n*m, rank) = (4, 7)
    # V shape: (m*p, rank) = (4, 7)
    # W shape: (p*n, rank) = (4, 7)
    
    # Indices for U (A): 0:00, 1:01, 2:10, 3:11
    # Indices for V (B): 0:00, 1:01, 2:10, 3:11
    # Indices for W (C): W[k*n+i,r] -> 0:00, 1:10, 2:01, 3:11
    
    # Define the 7 bilinear forms M1..M7
    # U_vec is length 4 (A00, A01, A10, A11)
    # V_vec is length 4 (B00, B01, B10, B11)
    
    u_vecs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    v_vecs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Construct U and V matrices
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    
    for k in range(rank):
        for i in range(n*m):
            U[i][k] = u_vecs[k][i]
        for j in range(m*p):
            V[j][k] = v_vecs[k][j]
            
    # Construct W matrix
    # W[r, k] = coefficient of M[k] in C[r]
    # C[0,0] (idx 0) = M1 + M4 - M5 + M7
    # C[1,0] (idx 1) = M2 + M4
    # C[0,1] (idx 2) = M3 + M5
    # C[1,1] (idx 3) = M1 - M2 + M3 + M6
    
    W = [[0]*rank for _ in range(p*n)]
    
    # idx 0: C[0,0]
    W[0][0] = 1
    W[0][3] = 1
    W[0][4] = -1
    W[0][6] = 1
    
    # idx 1: C[1,0]
    W[1][1] = 1
    W[1][3] = 1
    
    # idx 2: C[0,1]
    W[2][2] = 1
    W[2][4] = 1
    
    # idx 3: C[1,1]
    W[3][0] = 1
    W[3][1] = -1
    W[3][2] = 1
    W[3][5] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
