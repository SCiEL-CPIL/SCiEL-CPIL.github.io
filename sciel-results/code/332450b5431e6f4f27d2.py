"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using recursive Strassen
    # 4 blocks of k (size 2), 2 blocks of j (size 2)
    # Each block pair (s, t) uses 7 Strassen products
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Coefficients for the 7 Strassen products
    # Indices: (i', j') for A, (j', k') for B, (i', k') for C
    # A coeffs: [A00, A01, A10, A11]
    U_coeffs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # B coeffs: [B00, B01, B10, B11]
    V_coeffs = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [-1, 0, 1, 0], # M3
        [0, -1, 0, 1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    # C coeffs: [C00, C01, C10, C11]
    W_coeffs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 0, 1],  # M6
        [-1, 0, 0, -1] # M7
    ]
    
    col = 0
    for s in range(2):      # Block index for j (0, 1)
        for t in range(4):  # Block index for k (0, 1, 2, 3)
            for k_idx in range(7):
                # U part
                # A indices: i in {0,1}, j in {2s, 2s+1}
                # U index: i * 4 + j
                u_c = U_coeffs[k_idx]
                # A00 -> i=0, j=2s+0
                U[0 * 4 + (2 * s + 0)][col] = u_c[0]
                # A01 -> i=0, j=2s+1
                U[0 * 4 + (2 * s + 1)][col] = u_c[1]
                # A10 -> i=1, j=2s+0
                U[1 * 4 + (2 * s + 0)][col] = u_c[2]
                # A11 -> i=1, j=2s+1
                U[1 * 4 + (2 * s + 1)][col] = u_c[3]
                
                # V part
                # B indices: j in {2s, 2s+1}, k in {2t, 2t+1}
                # V index: j * 8 + k
                v_c = V_coeffs[k_idx]
                # B00 -> j=2s+0, k=2t+0
                V[(2 * s + 0) * 8 + (2 * t + 0)][col] = v_c[0]
                # B01 -> j=2s+0, k=2t+1
                V[(2 * s + 0) * 8 + (2 * t + 1)][col] = v_c[1]
                # B10 -> j=2s+1, k=2t+0
                V[(2 * s + 1) * 8 + (2 * t + 0)][col] = v_c[2]
                # B11 -> j=2s+1, k=2t+1
                V[(2 * s + 1) * 8 + (2 * t + 1)][col] = v_c[3]
                
                # W part
                # C indices: i in {0,1}, k in {2t, 2t+1}
                # W index: k * 2 + i (Trace convention W[k*n+i])
                w_c = W_coeffs[k_idx]
                # C00 -> i=0, k=2t+0
                W[(2 * t + 0) * 2 + 0][col] = w_c[0]
                # C01 -> i=0, k=2t+1
                W[(2 * t + 1) * 2 + 0][col] = w_c[1]
                # C10 -> i=1, k=2t+0
                W[(2 * t + 0) * 2 + 1][col] = w_c[2]
                # C11 -> i=1, k=2t+1
                W[(2 * t + 1) * 2 + 1][col] = w_c[3]
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
