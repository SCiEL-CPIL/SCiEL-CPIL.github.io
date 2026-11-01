"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 using 4 copies of Strassen (rank 7) for 2x2x2
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10+B11)
    # M4: A11(B00-B10)
    # M5: (A00+A10)B11
    # M6: (A11+A01)(B00+B01)
    # M7: (A00-A01)(B01+B11)
    
    # Coefficients for A (U)
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 0, 1, 0], # M5
        [0, 1, 0, 1], # M6
        [1, -1, 0, 0] # M7
    ]
    
    # Coefficients for B (V)
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, 1], # M3
        [1, 0, -1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 1, 0, 1]  # M7
    ]
    
    # Coefficients for C (W)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_coeffs = [
        [1, 0, 0, 1], # M1
        [-1, 0, 1, -1],# M2
        [-1, 1, 0, 1], # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]
    
    for block in range(4):
        k_offset = block * 2
        for t in range(7):
            col = block * 7 + t
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    u_idx = i * m + j
                    U[u_idx][col] = A_coeffs[t][i * 2 + j]
                    
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    k_global = k_offset + k_local
                    v_idx = j * p + k_global
                    V[v_idx][col] = B_coeffs[t][j * 2 + k_local]
                    
            # Fill W
            for k_local in range(2):
                for i in range(2):
                    k_global = k_offset + k_local
                    w_idx = k_global * n + i
                    W[w_idx][col] = C_coeffs[t][k_local * 2 + i]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
