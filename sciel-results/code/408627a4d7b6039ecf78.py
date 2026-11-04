import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 2x2 Strassen Decomposition
    # U2, V2, W2: shape (4, 7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # t=0: M1 = (a+d)(e+h)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # t=1: M2 = (c+d)e
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # t=2: M3 = a(f-h)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # t=3: M4 = d(g-e)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # t=4: M5 = (a+b)h
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # t=5: M6 = (c-a)(e+f)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # t=6: M7 = (b-d)(g+h)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # 3x3 Schoolbook Decomposition
    # U3, V3, W3: shape (9, 27)
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                t3 = i3 * 9 + j3 * 3 + k3
                r_u3 = i3 * 3 + j3
                r_v3 = j3 * 3 + k3
                r_w3 = k3 * 3 + i3
                U3[r_u3, t3] = 1
                V3[r_v3, t3] = 1
                W3[r_w3, t3] = 1

    # Construct index mappings for 6x6x6
    # Rows of U are indexed by i*6 + j
    # i = 2*i3 + i2
    # j = 2*j3 + j2
    # U row index in 2x2 factor: i2*2 + j2
    # U row index in 3x3 factor: i3*3 + j3
    
    i_vals = np.arange(6)
    j_vals = np.arange(6)
    I, J = np.meshgrid(i_vals, j_vals, indexing='ij')
    
    row_indices = (I * 6 + J).flatten() # 0..35
    
    i2 = (I % 2).flatten()
    i3 = (I // 2).flatten()
    j2 = (J % 2).flatten()
    j3 = (J // 2).flatten()
    
    idx_u2 = i2 * 2 + j2
    idx_u3 = i3 * 3 + j3
    
    # For V: rows j*6 + k
    K, L = np.meshgrid(i_vals, j_vals, indexing='ij') # Reusing I, J but renaming for clarity in logic
    # Actually V rows are j*6+k
    J_V, K_V = np.meshgrid(j_vals, np.arange(6), indexing='ij')
    row_indices_v = (J_V * 6 + K_V).flatten()
    j2_v = (J_V % 2).flatten()
    j3_v = (J_V // 2).flatten()
    k2_v = (K_V % 2).flatten()
    k3_v = (K_V // 2).flatten()
    
    idx_v2 = j2_v * 2 + k2_v
    idx_v3 = j3_v * 3 + k3_v
    
    # For W: rows k*6 + i
    K_W, I_W = np.meshgrid(np.arange(6), i_vals, indexing='ij')
    row_indices_w = (K_W * 6 + I_W).flatten()
    k2_w = (K_W % 2).flatten()
    k3_w = (K_W // 2).flatten()
    i2_w = (I_W % 2).flatten()
    i3_w = (I_W // 2).flatten()
    
    idx_w2 = k2_w * 2 + i2_w
    idx_w3 = k3_w * 3 + i3_w
    
    # Vectorized Kronecker-like construction
    # U[rows, t2*27+t3] = U2[idx_u2[rows], t2] * U3[idx_u3[rows], t3]
    
    # Select rows for U2 and U3
    U2_sel = U2[idx_u2, :] # shape (36, 7)
    U3_sel = U3[idx_u3, :] # shape (36, 27)
    
    # Outer product over the factor dimensions for each row
    # U2_sel[:, :, None] has shape (36, 7, 1)
    # U3_sel[:, None, :] has shape (36, 1, 27)
    # Product has shape (36, 7, 27)
    U_tensor = U2_sel[:, :, np.newaxis] * U3_sel[:, np.newaxis, :]
    U = U_tensor.reshape(36, 189)
    
    V2_sel = V2[idx_v2, :] # shape (36, 7)
    V3_sel = V3[idx_v3, :] # shape (36, 27)
    V_tensor = V2_sel[:, :, np.newaxis] * V3_sel[:, np.newaxis, :]
    V = V_tensor.reshape(36, 189)
    
    W2_sel = W2[idx_w2, :] # shape (36, 7)
    W3_sel = W3[idx_w3, :] # shape (36, 27)
    W_tensor = W2_sel[:, :, np.newaxis] * W3_sel[:, np.newaxis, :]
    W = W_tensor.reshape(36, 189)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
