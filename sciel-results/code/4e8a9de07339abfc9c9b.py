import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen rank-7 decomposition for 2x2 matrix multiplication
    # U2: rows A_ij, cols M_k. Index r = 2*i + j
    # V2: rows B_jk, cols M_k. Index r = 2*j + k
    # W2: rows C_ik, cols M_k. Index r = 2*i + k
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, -1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Kronecker product for 4x4
    # U4_kron = U2 (x) U2
    # V4_kron = V2 (x) V2
    # W4_kron = W2 (x) W2
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    rank = 49
    
    # Permutation matrices to adjust index ordering
    # U: Desired index 4*i + j. Kron index 8*i1 + 4*j1 + 2*i0 + j0.
    # i = 2*i1 + i0, j = 2*j1 + j0.
    P_U = np.zeros((16, 16), dtype=int)
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2*i1 + i0
                    j = 2*j1 + j0
                    r_des = 4*i + j
                    r_kron = 8*i1 + 4*j1 + 2*i0 + j0
                    P_U[r_des, r_kron] = 1
    
    # V: Desired index 4*j + k. Kron index 8*j1 + 4*k1 + 2*j0 + k0.
    # j = 2*j1 + j0, k = 2*k1 + k0.
    P_V = np.zeros((16, 16), dtype=int)
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2*j1 + j0
                    k = 2*k1 + k0
                    r_des = 4*j + k
                    r_kron = 8*j1 + 4*k1 + 2*j0 + k0
                    P_V[r_des, r_kron] = 1
    
    # W: Desired index 4*k + i. Kron index 8*i1 + 4*k1 + 2*i0 + k0.
    # Note: In W2, index was 2*i + k.
    # Kron index from W2 (x) W2:
    # Row index r_kron = 4*r2_1 + r2_0 = 4(2*i1 + k1) + (2*i0 + k0) = 8*i1 + 4*k1 + 2*i0 + k0.
    # Desired index 4*k + i.
    # k = 2*k1 + k0, i = 2*i1 + i0.
    P_W = np.zeros((16, 16), dtype=int)
    for i1 in range(2):
        for k1 in range(2):
            for i0 in range(2):
                for k0 in range(2):
                    i = 2*i1 + i0
                    k = 2*k1 + k0
                    r_des = 4*k + i
                    r_kron = 8*i1 + 4*k1 + 2*i0 + k0
                    P_W[r_des, r_kron] = 1
    
    U_final = P_U @ U4_kron
    V_final = P_V @ V4_kron
    W_final = P_W @ W4_kron
    
    # Convert to list of lists for output
    U_out = U_final.tolist()
    V_out = V_final.tolist()
    W_out = W_final.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
