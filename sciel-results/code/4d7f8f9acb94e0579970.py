import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 4, 4, 7
    
    # 1. Construct 4x4x4 Strassen decomposition (Rank 49)
    # 2x2 Strassen
    U2 = [[1, 0, 0, 1],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 1, 1]] # This is wrong, let's derive properly
    
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U (A coeffs), V (B coeffs), W (C coeffs)
    # Indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V Indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W Indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    U2_list = []
    V2_list = []
    W2_list = []
    
    # M1
    U2_list.append([1, 0, 0, 1])
    V2_list.append([1, 0, 0, 1])
    W2_list.append([1, 0, 0, 1]) # C00 + C11
    
    # M2
    U2_list.append([0, 0, 1, 1])
    V2_list.append([1, 0, 0, 0])
    W2_list.append([0, 0, 1, 0]) # C10
    
    # M3
    U2_list.append([1, 0, 0, 0])
    V2_list.append([0, 1, 0, -1])
    W2_list.append([0, 1, 0, 0]) # C01
    
    # M4
    U2_list.append([0, 0, 0, 1])
    V2_list.append([0, 0, 1, -1])
    W2_list.append([1, 0, 0, 0]) # C00
    
    # M5
    U2_list.append([1, 1, 0, 0])
    V2_list.append([0, 0, 0, 1])
    W2_list.append([-1, 0, 0, 0]) # -C00
    
    # M6
    U2_list.append([-1, 0, 1, 0])
    V2_list.append([1, 1, 0, 0])
    W2_list.append([0, 0, 0, 1]) # C11
    
    # M7
    U2_list.append([0, 1, 0, -1])
    V2_list.append([0, 0, 1, 1])
    W2_list.append([0, 1, 0, 0]) # C01
    
    U2 = np.array(U2_list).T # Shape 4x7
    V2 = np.array(V2_list).T # Shape 4x7
    W2 = np.array(W2_list).T # Shape 4x7
    
    # Kronecker Product for 4x4x4
    # U4 = U2 kron U2, with row permutation
    # V4 = V2 kron V2, with col permutation
    # W4 = W2 kron W2, with col permutation
    
    U4_kron = np.kron(U2, U2) # Shape 16x49
    V4_kron = np.kron(V2, V2) # Shape 16x49
    W4_kron = np.kron(W2, W2) # Shape 16x49
    
    # Permutation for U (Rows)
    # Kron Row: (i1*2+j1)*4 + (i2*2+j2)
    # Target Row: (i1*2+i2)*4 + (j1*2+j2)
    perm_U = []
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    r_kron = (i1*2 + j1)*4 + (i2*2 + j2)
                    r_target = (i1*2 + i2)*4 + (j1*2 + j2)
                    perm_U.append(r_kron)
    U4 = U4_kron[perm_U, :]
    
    # Permutation for V (Cols)
    # Kron Col: (j1*2+k1)*4 + (j2*2+k2)
    # Target Col: (j1*2+j2)*4 + (k1*2+k2)
    perm_V = []
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    c_kron = (j1*2 + k1)*4 + (j2*2 + k2)
                    c_target = (j1*2 + j2)*4 + (k1*2 + k2)
                    perm_V.append(c_kron)
    V4 = V4_kron[:, perm_V]
    
    # Permutation for W (Cols)
    # Kron Col: (k1*2+i1)*4 + (k2*2+i2)
    # Target Col: (k1*2+k2)*4 + (i1*2+i2)
    perm_W = []
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    c_kron = (k1*2 + i1)*4 + (k2*2 + i2)
                    c_target = (k1*2 + k2)*4 + (i1*2 + i2)
                    perm_W.append(c_kron)
    W4 = W4_kron[:, perm_W]
    
    # 2. Construct 4x4x3 Schoolbook decomposition (Rank 48)
    # This part handles k in [4, 7)
    rank_sch = 4 * 4 * 3
    U_sch = np.zeros((16, rank_sch))
    V_sch = np.zeros((28, rank_sch)) # Full V size
    W_sch = np.zeros((28, rank_sch)) # Full W size
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k_local in range(3):
                k_global = k_local + 4
                # U index: i*4+j
                U_sch[i*4+j, col] = 1
                # V index: j*7+k_global
                V_sch[j*7+k_global, col] = 1
                # W index: k_global*4+i
                W_sch[k_global*4+i, col] = 1
                col += 1
                
    # 3. Combine
    U_total = np.hstack([U4, U_sch])
    V_total = np.hstack([V4, V_sch])
    W_total = np.hstack([W4, W_sch])
    
    # Convert to lists of lists with Fractions
    def to_frac_list(arr):
        return [[Fraction(int(x)) for x in row] for row in arr.tolist()]
        
    return {
        "U": to_frac_list(U_total),
        "V": to_frac_list(V_total),
        "W": to_frac_list(W_total)
    }
