import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 78
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1
        
    # Helper to add Strassen block
    # a_r0, a_r1, a_c0, a_c1 : A block rows/cols
    # b_r0, b_r1, b_c0, b_c1 : B block rows/cols
    # c_r0, c_r1, c_c0, c_c1 : C block rows/cols
    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        
        # A indices
        a00_idx = a_r0 * m + a_c0
        a01_idx = a_r0 * m + (a_c0 + 1)
        a10_idx = (a_r0 + 1) * m + a_c0
        a11_idx = (a_r0 + 1) * m + (a_c0 + 1)
        
        # B indices
        b00_idx = b_r0 * p + b_c0
        b01_idx = b_r0 * p + (b_c0 + 1)
        b10_idx = (b_r0 + 1) * p + b_c0
        b11_idx = (b_r0 + 1) * p + (b_c0 + 1)
        
        # C indices
        c00_idx = c_c0 * n + c_r0
        c01_idx = (c_c0 + 1) * n + c_r0
        c10_idx = c_c0 * n + (c_r0 + 1)
        c11_idx = (c_c0 + 1) * n + (c_r0 + 1)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[a00_idx, col] = 1
        U[a11_idx, col] = 1
        V[b00_idx, col] = 1
        V[b11_idx, col] = 1
        W[c00_idx, col] = 1
        W[c11_idx, col] = 1
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10_idx, col] = 1
        U[a11_idx, col] = 1
        V[b00_idx, col] = 1
        W[c10_idx, col] = 1
        W[c11_idx, col] = -1
        col += 1
        
        # M3 = A00(B10 - B11)
        U[a00_idx, col] = 1
        V[b10_idx, col] = 1
        V[b11_idx, col] = -1
        W[c01_idx, col] = 1
        W[c10_idx, col] = 1
        col += 1
        
        # M4 = A11(B01 - B00)
        U[a11_idx, col] = 1
        V[b01_idx, col] = 1
        V[b00_idx, col] = -1
        W[c00_idx, col] = 1
        W[c10_idx, col] = 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00_idx, col] = 1
        U[a01_idx, col] = 1
        V[b11_idx, col] = 1
        W[c00_idx, col] = -1
        W[c01_idx, col] = 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B10)
        U[a10_idx, col] = 1
        U[a00_idx, col] = -1
        V[b00_idx, col] = 1
        V[b10_idx, col] = 1
        W[c11_idx, col] = 1
        col += 1
        
        # M7 = (A01 - A11)(B01 + B11)
        U[a01_idx, col] = 1
        U[a11_idx, col] = -1
        V[b01_idx, col] = 1
        V[b11_idx, col] = 1
        W[c00_idx, col] = 1
        W[c01_idx, col] = 1
        col += 1

    # 6 Strassen blocks
    # 1. A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
    add_strassen(0, 0, 0, 0, 0, 0)
    # 2. A[0:2, 2:4], B[2:4, 0:2], C[0:2, 0:2]
    add_strassen(0, 2, 2, 0, 0, 0)
    # 3. A[0:2, 0:2], B[0:2, 2:4], C[0:2, 2:4]
    add_strassen(0, 0, 0, 2, 0, 2)
    # 4. A[0:2, 2:4], B[2:4, 2:4], C[0:2, 2:4]
    add_strassen(0, 2, 2, 2, 0, 2)
    # 5. A[0:2, 0:2], B[0:2, 4:6], C[0:2, 4:6]
    add_strassen(0, 0, 0, 4, 0, 4)
    # 6. A[0:2, 2:4], B[2:4, 4:6], C[0:2, 4:6]
    add_strassen(0, 2, 2, 4, 0, 4)
    
    # Remaining schoolbook blocks
    # C[2:3, 0:2] : A[2:3, 0:2] * B[0:2, 0:2] + A[2:3, 2:4] * B[2:4, 0:2]
    for i in range(2, 3):
        for j in range(0, 2):
            for k in range(0, 2):
                add_schoolbook(i, j, k)
        for j in range(2, 4):
            for k in range(0, 2):
                add_schoolbook(i, j, k)
                
    # C[2:3, 2:4]
    for i in range(2, 3):
        for j in range(0, 2):
            for k in range(2, 4):
                add_schoolbook(i, j, k)
        for j in range(2, 4):
            for k in range(2, 4):
                add_schoolbook(i, j, k)
                
    # C[2:3, 4:6]
    for i in range(2, 3):
        for j in range(0, 2):
            for k in range(4, 6):
                add_schoolbook(i, j, k)
        for j in range(2, 4):
            for k in range(4, 6):
                add_schoolbook(i, j, k)
                
    # C[0:2, 6:7]
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(6, 7):
                add_schoolbook(i, j, k)
        for j in range(2, 4):
            for k in range(6, 7):
                add_schoolbook(i, j, k)
                
    # C[2:3, 6:7]
    for i in range(2, 3):
        for j in range(0, 2):
            for k in range(6, 7):
                add_schoolbook(i, j, k)
        for j in range(2, 4):
            for k in range(6, 7):
                add_schoolbook(i, j, k)

    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
