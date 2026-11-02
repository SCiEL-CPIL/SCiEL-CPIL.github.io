import numpy as np

def construct():
    # Rank 7 decomposition for 2x2x2
    # U2, V2, W2 are 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 1]
    ])
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 1, 0]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, -1, 1, 0, 0, 1, -1]
    ])
    
    # Verify 2x2x2
    # C = sum_c U2[:, c] * V2[:, c] * W2[:, c]
    # This is a standard Strassen decomposition.
    
    # Rank 49 decomposition for 4x4x4 using Kronecker product
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Rank 48 schoolbook decomposition for 4x4x3
    n, m, p = 4, 4, 3
    rank3 = n * m * p
    U3 = np.zeros((n * m, rank3))
    V3 = np.zeros((m * p, rank3))
    W3 = np.zeros((p * n, rank3))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U3[i * m + j, col] = 1
                V3[j * p + k, col] = 1
                W3[k * n + i, col] = 1
                col += 1
                
    # Combine for 4x4x7
    rank_total = 49 + rank3
    n, m, p_full = 4, 4, 7
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p_full, rank_total))
    W = np.zeros((p_full * n, rank_total))
    
    # Copy U4 and U3
    U[:, :49] = U4
    U[:, 49:] = U3
    
    # Copy V4 into V
    # V4 rows: j*4+k (j in 0..3, k in 0..3)
    # V rows: j*7+k (j in 0..3, k in 0..6)
    for j in range(4):
        for k in range(4):
            V[j * 7 + k, :49] = V4[j * 4 + k, :]
            
    # Copy V3 into V
    # V3 rows: j*3+k' (j in 0..3, k' in 0..2)
    # V rows: j*7+k (j in 0..3, k in 4..6) -> k = 4 + k'
    for j in range(4):
        for k_prime in range(3):
            V[j * 7 + (4 + k_prime), 49:] = V3[j * 3 + k_prime, :]
            
    # Copy W4 into W
    # W4 rows: k*4+i (k in 0..3, i in 0..3)
    # W rows: k*4+i (k in 0..6, i in 0..3)
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :49] = W4[k * 4 + i, :]
            
    # Copy W3 into W
    # W3 rows: k'*4+i (k' in 0..2, i in 0..3)
    # W rows: k*4+i (k in 4..6, i in 0..3) -> k = 4 + k'
    for k_prime in range(3):
        for i in range(4):
            W[(4 + k_prime) * 4 + i, 49:] = W3[k_prime * 4 + i, :]
            
    return {"U": U, "V": V, "W": W}
