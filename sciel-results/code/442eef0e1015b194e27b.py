import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1 # C10
    
    # M3 = A00(B01-B11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1 # C01
    
    # M4 = A11(B10-B00)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1 # C00, C10
    
    # M5 = (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1 # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1 # C00

    # Verify Strassen
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Kronecker product for 4x4x4
    # U_444 = U_s (x) U_s
    U_444 = np.kron(U_s, U_s)
    V_444 = np.kron(V_s, V_s)
    W_444 = np.kron(W_s, W_s)
    
    # Dimensions for 4x4x8
    # U: 16 x 98
    # V: 32 x 98
    # W: 32 x 98
    
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # Block 1 (k=0..3)
    U[:, 0:49] = U_444
    # V rows: j*8+k. For k=0..3, rows are j*8+k.
    # V_444 rows: j*4+k.
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V_444[j*4 + k, :]
            
    # W rows: k*4+i. For k=0..3, rows are k*4+i.
    # W_444 rows: k*4+i.
    W[0:16, 0:49] = W_444
    
    # Block 2 (k=4..7)
    U[:, 49:98] = U_444
    for j in range(4):
        for k in range(4):
            V[j*8 + (k+4), 49:98] = V_444[j*4 + k, :]
            
    W[16:32, 49:98] = W_444

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
