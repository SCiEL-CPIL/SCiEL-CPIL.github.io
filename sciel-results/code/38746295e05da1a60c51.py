import numpy as np

def strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication.
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def strassen_4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using Kronecker product of 2x2.
    U: (16, 49)
    V: (16, 49)
    W: (16, 49)
    """
    U2, V2, W2 = strassen_2x2()
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    return U4, V4, W4

def trivial_4x4x1():
    """
    Returns U, V, W for 4x4x1 matrix multiplication (matrix-vector).
    U: (16, 16)
    V: (4, 16)
    W: (4, 16)
    """
    U = np.zeros((16, 16), dtype=float)
    V = np.zeros((4, 16), dtype=float)
    W = np.zeros((4, 16), dtype=float)
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U[i * 4 + j, r] = 1.0
            V[j, r] = 1.0
            W[i, r] = 1.0
            
    return U, V, W

def construct():
    """
    Construct rank-minimal exact tensor decomposition for 4x4x9 matrix multiplication.
    Rank = 49 (first 4 cols) + 16 (5th col) + 49 (last 4 cols) = 114.
    """
    # Total dimensions
    n = 4
    m = 4
    p = 9
    rank = 114
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Get components
    U_s4, V_s4, W_s4 = strassen_4x4()
    U_t1, V_t1, W_t1 = trivial_4x4x1()
    
    # Block 1: Columns 0-3 of B, Columns 0-3 of C
    # U: 16x49 -> U[:, 0:49]
    U[:, 0:49] = U_s4
    
    # V: 16x49 -> V[:, 0:49]
    # V_s4 indices: row j (0-3), col k (0-3). Total 16 rows.
    # V global indices: row j (0-3), col k (0-3). Index = j*9 + k.
    for j in range(4):
        for k in range(4):
            v_local_idx = j * 4 + k
            v_global_idx = j * 9 + k
            V[v_global_idx, 0:49] = V_s4[v_local_idx, :]
            
    # W: 16x49 -> W[:, 0:49]
    # W_s4 indices: row i (0-3), col k (0-3). Total 16 rows.
    # W global indices: row i (0-3), col k (0-3). Index = i*9 + k.
    for i in range(4):
        for k in range(4):
            w_local_idx = i * 4 + k
            w_global_idx = i * 9 + k
            W[w_global_idx, 0:49] = W_s4[w_local_idx, :]
            
    # Block 2: Column 4 of B, Column 4 of C
    # U: 16x16 -> U[:, 49:65]
    U[:, 49:65] = U_t1
    
    # V: 4x16 -> V[:, 49:65]
    # V_t1 indices: row j (0-3).
    # V global indices: row j (0-3), col 4. Index = j*9 + 4.
    for j in range(4):
        v_global_idx = j * 9 + 4
        V[v_global_idx, 49:65] = V_t1[j, :]
        
    # W: 4x16 -> W[:, 49:65]
    # W_t1 indices: row i (0-3).
    # W global indices: row i (0-3), col 4. Index = i*9 + 4.
    for i in range(4):
        w_global_idx = i * 9 + 4
        W[w_global_idx, 49:65] = W_t1[i, :]
        
    # Block 3: Columns 5-8 of B, Columns 5-8 of C
    # U: 16x49 -> U[:, 65:114]
    U[:, 65:114] = U_s4
    
    # V: 16x49 -> V[:, 65:114]
    # V_s4 indices: row j (0-3), col k (0-3).
    # V global indices: row j (0-3), col 5+k. Index = j*9 + (5+k).
    for j in range(4):
        for k in range(4):
            v_local_idx = j * 4 + k
            v_global_idx = j * 9 + (5 + k)
            V[v_global_idx, 65:114] = V_s4[v_local_idx, :]
            
    # W: 16x49 -> W[:, 65:114]
    # W_s4 indices: row i (0-3), col k (0-3).
    # W global indices: row i (0-3), col 5+k. Index = i*9 + (5+k).
    for i in range(4):
        for k in range(4):
            w_local_idx = i * 4 + k
            w_global_idx = i * 9 + (5 + k)
            W[w_global_idx, 65:114] = W_s4[w_local_idx, :]
            
    return {'U': U, 'V': V, 'W': W}
