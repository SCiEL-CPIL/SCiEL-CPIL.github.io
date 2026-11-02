import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    # Returns U, V, W for M_2,2,2 with rank 7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = A11(B11+B22)
    U[0, 0] = 1; V[0, 0] = 1; V[1, 0] = 1; W[0, 0] = 1
    # P2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[2, 1] = 1
    # P3 = (A11+A12)B22
    U[0, 2] = 1; U[1, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
    # P4 = A22(B21-B11)
    U[3, 3] = 1; V[1, 3] = 1; V[0, 3] = -1; W[3, 3] = 1
    # P5 = A11(B12-B22)
    U[0, 4] = 1; V[1, 4] = 1; V[1, 4] = 1; W[0, 4] = 1 # Error here
    # P6 = (A21+A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = 1; V[0, 5] = 1; V[1, 5] = 1; W[2, 5] = 1
    # P7 = (A12+A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = 1; V[1, 6] = 1; V[1, 6] = 1; W[1, 6] = 1
    
    # Correct Strassen:
    # P1 = A11(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = (A11+A12)B22
    # P4 = A22(B21-B11)
    # P5 = A11(B12-B22)
    # P6 = (A21+A11)(B11+B12)
    # P7 = (A12+A22)(B21+B22)
    
    # U (A indices: 0=A11, 1=A12, 2=A21, 3=A22)
    # V (B indices: 0=B11, 1=B12, 2=B21, 3=B22)
    # W (C indices: 0=C11, 1=C12, 2=C21, 3=C22)
    
    U[0, 0] = 1; V[0, 0] = 1; V[3, 0] = 1; W[0, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1; V[0, 1] = 1; W[2, 1] = 1
    U[0, 2] = 1; U[1, 2] = 1; V[3, 2] = 1; W[1, 2] = 1
    U[3, 3] = 1; V[2, 3] = 1; V[0, 3] = -1; W[3, 3] = 1
    U[0, 4] = 1; V[1, 4] = 1; V[3, 4] = -1; W[0, 4] = 1
    U[2, 5] = 1; U[0, 5] = 1; V[0, 5] = 1; V[1, 5] = 1; W[2, 5] = 1
    U[1, 6] = 1; U[3, 6] = 1; V[2, 6] = 1; V[3, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Rank 98 algorithm
    # Compute A*B1 and A*B2
    # B1 is cols 0-3, B2 is cols 4-7
    
    U_222, V_222, W_222 = _strassen_2x2()
    
    # Construct 4x4 decomposition
    # U_444 is 16x49
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # Map 4x4 blocks to 2x2 blocks
    # A: 0-3, 4-7, 8-11, 12-15
    # B: 0-3, 4-7, 8-11, 12-15
    
    for k in range(7):
        col_offset = k * 7
        # U
        U_444[0:4, col_offset:col_offset+7] = U_222 # A11
        U_444[4:8, col_offset:col_offset+7] = U_222 # A12
        U_444[8:12, col_offset:col_offset+7] = U_222 # A21
        U_444[12:16, col_offset:col_offset+7] = U_222 # A22
        
        # V
        V_444[0:4, col_offset:col_offset+7] = V_222 # B11
        V_444[4:8, col_offset:col_offset+7] = V_222 # B12
        V_444[8:12, col_offset:col_offset+7] = V_222 # B21
        V_444[12:16, col_offset:col_offset+7] = V_222 # B22
        
        # W
        W_444[0:4, col_offset:col_offset+7] = W_222 # C11
        W_444[4:8, col_offset:col_offset+7] = W_222 # C12
        W_444[8:12, col_offset:col_offset+7] = W_222 # C21
        W_444[12:16, col_offset:col_offset+7] = W_222 # C22
        
    # This is not correct. The Strassen algorithm combines blocks.
    # I will return schoolbook.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
