import numpy as np

def construct():
    n, m, p = 4, 4, 4
    
    # 2x2 Strassen matrices
    # U2, V2, W2 are 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    # W2 rows permuted to match 2k+i order: 0:C00, 1:C10, 2:C01, 3:C11
    # Cols 1..7
    W2 = np.array([
        [1, 0, -1, 1, -1, 0, 1], # C00
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [1, -1, 0, 0, 0, 1, 0]   # C11
    ])
    
    # Kronecker products
    U_kron = np.kron(U2, U2)
    V_kron = np.kron(V2, V2)
    W_kron = np.kron(W2, W2)
    
    # Permutation for U and V
    # idx_std = 8*I + 4*i + 2*J + j
    # k_kron  = 8*I + 4*J + 2*i + j
    # We need U_std[idx_std] = U_kron[k_kron]
    # So we create an array of k_kron for each idx_std
    
    perm_UV = []
    for idx_std in range(16):
        # Extract I, i, J, j from idx_std
        I = idx_std // 8
        rem1 = idx_std % 8
        i = rem1 // 4
        rem2 = rem1 % 4
        J = rem2 // 2
        j = rem2 % 2
        
        # Calculate k_kron
        k_kron = 8 * I + 4 * J + 2 * i + j
        perm_UV.append(k_kron)
        
    U = U_kron[perm_UV, :]
    V = V_kron[perm_UV, :]
    
    # Permutation for W
    # idx_std = 8*k1 + 4*k2 + 2*i1 + i2
    # k_kron  = 8*k1 + 4*i1 + 2*k2 + i2
    perm_W = []
    for idx_std in range(16):
        # Extract k1, k2, i1, i2 from idx_std
        k1 = idx_std // 8
        rem1 = idx_std % 8
        k2 = rem1 // 4
        rem2 = rem1 % 4
        i1 = rem2 // 2
        i2 = rem2 % 2
        
        # Calculate k_kron
        k_kron = 8 * k1 + 4 * i1 + 2 * k2 + i2
        perm_W.append(k_kron)
        
    W = W_kron[perm_W, :]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
