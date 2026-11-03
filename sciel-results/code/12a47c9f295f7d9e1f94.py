import numpy as np

def construct():
    n, m, p = 6, 6, 6
    
    # 1. Construct 3x3 schoolbook decomposition (Rank 27)
    n3 = 3
    rank3 = n3 * n3 * n3
    U33 = np.zeros((n3*n3, rank3))
    V33 = np.zeros((n3*n3, rank3))
    W33 = np.zeros((n3*n3, rank3))
    
    col = 0
    for i in range(n3):
        for j in range(n3):
            for k in range(n3):
                # A index: i * n3 + k
                # B index: k * n3 + j
                # C index: j * n3 + i
                U33[i * n3 + k, col] = 1
                V33[k * n3 + j, col] = 1
                W33[j * n3 + i, col] = 1
                col += 1
                
    # 2. Construct 2x2 Strassen decomposition (Rank 7)
    n2 = 2
    rank2 = 7
    U22 = np.zeros((n2*n2, rank2))
    V22 = np.zeros((n2*n2, rank2))
    W22 = np.zeros((n2*n2, rank2))
    
    # Strassen formulas
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices for 2x2:
    # 0: 11, 1: 12, 2: 21, 3: 22
    
    # Col 0: M1
    U22[0, 0] = 1; U22[3, 0] = 1
    V22[0, 0] = 1; V22[3, 0] = 1
    W22[0, 0] = 1; W22[3, 0] = 1
    
    # Col 1: M2
    U22[2, 1] = 1; U22[3, 1] = 1
    V22[0, 1] = 1
    W22[2, 1] = 1; W22[3, 1] = 1
    
    # Col 2: M3
    U22[0, 2] = 1
    V22[1, 2] = 1; V22[3, 2] = -1
    W22[1, 2] = 1; W22[3, 2] = 1
    
    # Col 3: M4
    U22[3, 3] = 1
    V22[2, 3] = 1; V22[0, 3] = -1
    W22[0, 3] = 1; W22[2, 3] = 1
    
    # Col 4: M5
    U22[0, 4] = 1; U22[1, 4] = 1
    V22[3, 4] = 1
    W22[1, 4] = 1; W22[3, 4] = 1
    
    # Col 5: M6
    U22[2, 5] = 1; U22[0, 5] = -1
    V22[0, 5] = 1; V22[1, 5] = 1
    W22[2, 5] = 1
    
    # Col 6: M7
    U22[1, 6] = 1; U22[3, 6] = -1
    V22[2, 6] = 1; V22[3, 6] = 1
    W22[3, 6] = 1
    
    # 3. Kronecker Product
    # U = U33 (Kron) U22
    # Shape: (9*4, 27*7) = (36, 189)
    U_kron = np.kron(U33, U22)
    V_kron = np.kron(V33, V22)
    W_kron = np.kron(W33, W22)
    
    # 4. Permute Rows
    # We need to map kron indices to actual indices.
    # Actual index for A: i * 6 + j
    # Kron index: i_b * 4 + i_e * ? 
    # Wait, np.kron(A, B) where A is IxJ, B is KxL.
    # Result is IK x JL.
    # Row r in result corresponds to i = r // K, k = r % K.
    # Value is A[i, :] * B[k, :]? No.
    # np.kron(A, B)[i*K + k, j*L + l] = A[i, j] * B[k, l].
    # So row index r = i * K + k.
    # Here A=U33 (9x27), B=U22 (4x7).
    # Row r = i_b * 4 + k_b.
    # i_b is index in U33 (0..8). k_b is index in U22 (0..3).
    # i_b corresponds to i3 * 3 + k3? No.
    # U33 rows are indexed by i3 * 3 + k3 (A index).
    # U22 rows are indexed by i2 * 2 + k2 (A index).
    # So r = (i3 * 3 + k3) * 4 + (i2 * 2 + k2) = i3*12 + k3*4 + i2*2 + k2.
    # Actual A index: (i3*2 + i2) * 6 + (k3*2 + k2) = i3*12 + i2*6 + k3*2 + k2.
    
    # We need a permutation P such that U_perm[r_actual, c] = U_kron[r_kron, c].
    # So we build a mapping from r_actual to r_kron.
    
    def get_permutation(n_block, n_elem, n_total):
        # n_block=3, n_elem=2, n_total=6
        # r_kron = i_b * (n_elem*n_elem) + k_b * n_elem + i_e * 2 + k_e ?
        # Let's re-verify np.kron structure.
        # A is I x J. B is K x L.
        # r = i * K + k.
        # i is index in A (0..I-1). k is index in B (0..K-1).
        # A index i = i_b * n_elem + i_e? No.
        # A is 3x3. i = i_row * 3 + i_col.
        # B is 2x2. k = k_row * 2 + k_col.
        # r_kron = i * 4 + k = (i_row * 3 + i_col) * 4 + (k_row * 2 + k_col)
        #        = i_row * 12 + i_col * 4 + k_row * 2 + k_col.
        # r_actual = (i_row * 2 + k_row) * 6 + (i_col * 2 + k_col)
        #          = i_row * 12 + k_row * 6 + i_col * 2 + k_col.
        
        mapping = {}
        for i_row in range(n_block):
            for i_col in range(n_block):
                for k_row in range(n_elem):
                    for k_col in range(n_elem):
                        r_kron = (i_row * n_block + i_col) * (n_elem * n_elem) + (k_row * n_elem + k_col)
                        r_actual = (i_row * n_elem + k_row) * n_total + (i_col * n_elem + k_col)
                        mapping[r_actual] = r_kron
        return mapping

    # For A (U)
    # i_row is row of block, i_col is col of block?
    # A index is i * n + j.
    # In 3x3, i is row, j is col?
    # U33 index was i * 3 + k. Here k is col.
    # So i_row = i, i_col = k.
    # In 2x2, i is row, j is col.
    # So k_row = i, k_col = j.
    
    map_U = get_permutation(3, 2, 6)
    map_V = get_permutation(3, 2, 6) # V index is j * n + k. j is row, k is col.
    map_W = get_permutation(3, 2, 6) # W index is k * n + i. k is row, i is col.
    
    U = np.zeros((36, 189))
    V = np.zeros((36, 189))
    W = np.zeros((36, 189))
    
    for r_actual in range(36):
        r_kron = map_U[r_actual]
        U[r_actual, :] = U_kron[r_kron, :]
        V[r_actual, :] = V_kron[map_V[r_actual], :]
        W[r_actual, :] = W_kron[map_W[r_actual], :]
        
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
