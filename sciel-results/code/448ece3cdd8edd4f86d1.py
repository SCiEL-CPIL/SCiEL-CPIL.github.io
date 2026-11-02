"""Matrix multiplication 2,2,16 - rank-minimal exact tensor decomposition."""

def construct():
    """
    Returns U, V, W factor matrices for M_{2,2,16} with rank 56.
    
    Uses 8 independent Strassen decompositions (rank 7 each) for the 8 blocks
    of 2 columns in the 16-column matrix.
    
    U: shape (4, 56) - input A (2x2)
    V: shape (32, 56) - input B (2x16)
    W: shape (32, 56) - output C (2x16)
    
    The Strassen decomposition for M_{2,2,2}:
    M1 = A11*(B11+B22)
    M2 = (A11+A22)*B22
    M3 = A21*(B11-B21)
    M4 = A22*(B22-B12)
    M5 = (A11+A12)*B21
    M6 = (A21-A22)*(B12+B22)
    M7 = (A11-A12)*(B11-B12)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    """
    import numpy as np
    
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7  # 56
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for U (A indices: 0=A11, 1=A12, 2=A21, 3=A22)
    # Each of the 7 Strassen products has specific A coefficients
    U_strassen = [
        [1, 0, 0, 0],  # M1: A11
        [1, 0, 0, 1],  # M2: A11+A22
        [0, 0, 1, 0],  # M3: A21
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11+A12
        [-1, 0, 1, 0], # M6: A21-A22
        [1, -1, 0, 0], # M7: A11-A12
    ]
    
    # Strassen coefficients for V (B indices: 0=B11, 1=B12, 2=B21, 3=B22)
    V_strassen = [
        [1, 0, 0, 1],  # M1: B11+B22
        [0, 0, 0, 1],  # M2: B22
        [1, 0, -1, 0], # M3: B11-B21
        [0, -1, 0, 1], # M4: B22-B12
        [0, 0, 1, 0],  # M5: B21
        [0, 1, 0, 1],  # M6: B12+B22
        [1, -1, 0, 0], # M7: B11-B12
    ]
    
    # Strassen coefficients for W (C indices: 0=C11, 1=C12, 2=C21, 3=C22)
    W_strassen = [
        [1, 0, 0, 1],  # M1: C11+M1, C22+M1
        [0, 0, 0, -1], # M2: C22-M2
        [0, 1, 0, 1],  # M3: C12+M3, C22+M3
        [1, 0, 1, 0],  # M4: C11+M4, C21+M4
        [-1, 1, 0, 0], # M5: C11-M5, C12+M5
        [0, 0, 0, 1],  # M6: C22+M6
        [1, 0, 0, 1],  # M7: C11+M7, C22+M7
    ]
    
    for block in range(num_blocks):
        for s in range(7):
            col = block * 7 + s
            
            # Set U column
            for i in range(4):
                U[i, col] = U_strassen[s][i]
            
            # Set V column (B rows: block*2 and block*2+1 for each of 2 rows)
            # V index: row*16 + col_in_block
            # For block b, the two B columns are 2*b and 2*b+1
            # V has 2 rows (for the 2x16 matrix), so V[row*16 + col_idx, col]
            b11_idx = 0 * p + (2 * block)      # B[0][2*block]
            b12_idx = 0 * p + (2 * block + 1)  # B[0][2*block+1]
            b21_idx = 1 * p + (2 * block)      # B[1][2*block]
            b22_idx = 1 * p + (2 * block + 1)  # B[1][2*block+1]
            
            V[b11_idx, col] = V_strassen[s][0]
            V[b12_idx, col] = V_strassen[s][1]
            V[b21_idx, col] = V_strassen[s][2]
            V[b22_idx, col] = V_strassen[s][3]
            
            # Set W column (C rows: output is 2x16, trace convention W[k*n+i, r])
            # C[i][k] where i is row (0,1) and k is column (0..15)
            # W index: k*n + i = k*2 + i
            c11_idx = (2 * block) * n + 0      # C[0][2*block]
            c12_idx = (2 * block + 1) * n + 0  # C[0][2*block+1]
            c21_idx = (2 * block) * n + 1      # C[1][2*block]
            c22_idx = (2 * block + 1) * n + 1  # C[1][2*block+1]
            
            W[c11_idx, col] = W_strassen[s][0]
            W[c12_idx, col] = W_strassen[s][1]
            W[c21_idx, col] = W_strassen[s][2]
            W[c22_idx, col] = W_strassen[s][3]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
