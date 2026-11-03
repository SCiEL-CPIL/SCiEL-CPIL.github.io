import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # Part 1: 4x4x4 multiplication using Strassen (Kronecker of 2x2x2 Strassen)
    # Rank 49
    # Part 2: 4x4x3 multiplication using Schoolbook
    # Rank 48
    # Total Rank 97
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen ---
    # The 2x2x2 Strassen algorithm uses 7 multiplications.
    # The 4x4x4 algorithm is the Kronecker product of the 2x2x2 algorithm with itself.
    # We need to define the U, V, W for the 2x2x2 Strassen first.
    
    # 2x2x2 Strassen Tensor
    # Inputs: A (2x2), B (2x2). Output C (2x2).
    # Indices i, j in {0,1} for A, j, k in {0,1} for B, k, i in {0,1} for C.
    # The 7 products are:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # We need to map these to the tensor indices.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Let's generate the U, V, W for the base 2x2x2 Strassen.
    # Base n=2, m=2, p=2.
    # U_base (4, 7), V_base (4, 7), W_base (4, 7)
    
    U_base = [[0]*7 for _ in range(4)]
    V_base = [[0]*7 for _ in range(4)]
    W_base = [[0]*7 for _ in range(4)]
    
    # M1
    # (A00 + A11) * (B00 + B11)
    # U: A00 (0), A11 (3) -> 1
    # V: B00 (0), B11 (3) -> 1
    # W: C00 (0) += M1, C01 (1) += M1, C10 (2) += M1, C11 (3) += M1
    # Wait, the contributions to C are:
    # C00 = M1 - M2 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    
    # Let's fill U_base, V_base, W_base carefully.
    
    # M1: (A00+A11)*(B00+B11)
    U_base[0][0] = 1; U_base[3][0] = 1
    V_base[0][0] = 1; V_base[3][0] = 1
    W_base[0][0] = 1; W_base[1][0] = 1; W_base[2][0] = 1; W_base[3][0] = 1
    # But we need to subtract in W.
    # W[k*n+i, c]
    # C00 (k=0,i=0) gets +M1
    # C01 (k=1,i=0) gets +M1? No.
    # Let's check Strassen formulas again.
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # c00 = M1 - M2 - M4 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M3 + M6 - M7
    
    # So for column 0 (M1):
    # U: a00, a11
    # V: b00, b11
    # W: c00 (+1), c01 (0), c10 (0), c11 (+1) -> Wait, c01 and c10 don't get M1?
    # My previous W_base fill was wrong.
    
    # Let's re-fill W_base.
    W_base = [[0]*7 for _ in range(4)]
    # Col 0 (M1)
    U_base[0][0] = 1; U_base[3][0] = 1
    V_base[0][0] = 1; V_base[3][0] = 1
    W_base[0][0] = 1  # c00
    W_base[3][0] = 1  # c11
    
    # Col 1 (M2)
    U_base[2][1] = 1; U_base[3][1] = 1 # a10, a11
    V_base[0][1] = 1 # b00
    W_base[2][1] = 1 # c10 (+M2)
    W_base[0][1] = -1 # c00 (-M2)
    
    # Col 2 (M3)
    U_base[0][2] = 1 # a00
    V_base[1][2] = 1; V_base[3][2] = -1 # b01, -b11
    W_base[1][2] = 1 # c01 (+M3)
    W_base[3][2] = -1 # c11 (-M3)
    
    # Col 3 (M4)
    U_base[3][3] = 1 # a11
    V_base[2][3] = 1; V_base[0][3] = -1 # b10, -b00
    W_base[2][3] = 1 # c10 (+M4)
    W_base[0][3] = -1 # c00 (-M4)
    
    # Col 4 (M5)
    U_base[0][4] = 1; U_base[1][4] = 1 # a00, a01
    V_base[3][4] = 1 # b11
    W_base[1][4] = 1 # c01 (+M5)
    
    # Col 5 (M6)
    U_base[2][5] = 1; U_base[0][5] = -1 # a10, -a00
    V_base[0][5] = 1; V_base[1][5] = 1 # b00, b01
    W_base[3][5] = 1 # c11 (+M6)
    
    # Col 6 (M7)
    U_base[1][6] = 1; U_base[3][6] = -1 # a01, -a11
    V_base[2][6] = 1; V_base[3][6] = 1 # b10, b11
    W_base[0][6] = 1 # c00 (+M7)
    W_base[3][6] = -1 # c11 (-M7)
    
    # Now, construct the 4x4x4 tensors U1, V1, W1 from U_base, V_base, W_base.
    # The Kronecker product structure:
    # Index i in 4x4 corresponds to (i1, i2) where i = 2*i1 + i2.
    # The tensor is sum over c1, c2 of U_base[i1, c1] * U_base[i2, c2] * ...
    # Actually, the rank is 7*7=49.
    # The columns of U1 are indexed by (c1, c2) where c1, c2 in 0..6.
    # col = c1 * 7 + c2.
    # U1[i, col] = U_base[i1, c1] * U_base[i2, c2]
    # V1[j, col] = V_base[j1, c1] * V_base[j2, c2]
    # W1[k, col] = W_base[k1, c1] * W_base[k2, c2]
    
    U1 = [[0]*49 for _ in range(16)]
    V1 = [[0]*49 for _ in range(16)] # Only first 16 rows of V (for B1)
    W1 = [[0]*49 for _ in range(16)] # Only first 16 rows of W (for C1)
    
    for c1 in range(7):
        for c2 in range(7):
            col = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2 * i1 + i2
                    U1[i][col] = U_base[i1][c1] * U_base[i2][c2]
            
            for j1 in range(2):
                for j2 in range(2):
                    j = 2 * j1 + j2
                    V1[j][col] = V_base[j1][c1] * V_base[j2][c2]
            
            for k1 in range(2):
                for k2 in range(2):
                    k = 2 * k1 + k2
                    W1[k][col] = W_base[k1][c1] * W_base[k2][c2]
    
    # Fill U, V, W for the first 49 columns
    for i in range(16):
        for c in range(49):
            U[i][c] = U1[i][c]
            
    for j in range(16):
        for c in range(49):
            V[j][c] = V1[j][c]
            
    for k in range(16):
        for c in range(49):
            W[k][c] = W1[k][c]
            
    # --- Part 2: 4x4x3 Schoolbook ---
    # Columns 49..96
    # Input B2 is columns 4,5,6 of B.
    # V index: j*7 + k. k in 4,5,6.
    # W index: k*4 + i. k in 4,5,6.
    
    col_offset = 49
    idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                k_abs = 4 + k
                U[i*4 + j][col_offset + idx] = 1
                V[j*7 + k_abs][col_offset + idx] = 1
                W[k_abs*4 + i][col_offset + idx] = 1
                idx += 1
                
    return {"U": U, "V": V, "W": W}
