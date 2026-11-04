import numpy as np

def construct():
    n, m, p = 6, 6, 6
    
    # 1. Construct Strassen decomposition for 2x2 (Rank 7)
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1 # C10, C11
    
    # M3
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1 # C01, C11
    
    # M4
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[2, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1 # C00, C10
    
    # M5
    U2[0, 4] = 1; U2[2, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1 # C00, C01
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[2, 5] = 1
    W2[3, 5] = 1 # C11
    
    # M7
    U2[3, 6] = 1; U2[1, 6] = -1
    V2[1, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C00

    # 2. Construct Schoolbook decomposition for 3x3 (Rank 27)
    # U3, V3, W3 are 9x27
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i * 3 + j, col] = 1
                V3[j * 3 + k, col] = 1
                W3[k * 3 + i, col] = 1
                col += 1

    # 3. Combine via Kronecker product
    # U_total = U3 (Kron) U2
    U_total = np.kron(U3, U2)
    V_total = np.kron(V3, V2)
    W_total = np.kron(W3, W2)
    
    # 4. Reorder indices to match required format
    # Required U index: (i_block * 2 + i_inner) * 6 + (j_block * 2 + j_inner)
    # Kronecker index: (i_block * 3 + j_block) * 4 + (i_inner * 2 + j_inner)
    
    def permute_indices(mat, n_outer, n_inner, total_dim):
        # mat shape: (n_outer*n_inner * n_inner? No)
        # mat is (N_outer * N_inner, Rank)
        # N_outer = 9 (3x3), N_inner = 4 (2x2)
        # Total rows = 36
        permuted = np.zeros_like(mat)
        for i_b in range(3):
            for j_b in range(3):
                idx_outer = i_b * 3 + j_b
                for i_i in range(2):
                    for j_i in range(2):
                        idx_inner = i_i * 2 + j_i
                        src_idx = idx_outer * 4 + idx_inner
                        dst_idx = (i_b * 2 + i_i) * 6 + (j_b * 2 + j_i)
                        permuted[dst_idx, :] = mat[src_idx, :]
        return permuted

    U_final = permute_indices(U_total, 3, 2, 36)
    V_final = permute_indices(V_total, 3, 2, 36)
    W_final = permute_indices(W_total, 3, 2, 36)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
