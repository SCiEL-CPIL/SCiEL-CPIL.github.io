"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # Rows: 0:A00, 1:A01, 2:A10, 3:A11
    U_coeffs = [
        [1, 0, 1, 0, 1, -1, 0], # M1..M7 for A00? No.
        # Let's list U[row][r]
        # r=0 (M1): a+d -> A00:1, A11:1
        # r=1 (M2): c+d -> A10:1, A11:1
        # r=2 (M3): a   -> A00:1
        # r=3 (M4): d   -> A11:1
        # r=4 (M5): a+b -> A00:1, A01:1
        # r=5 (M6): c-a -> A10:1, A00:-1
        # r=6 (M7): b-d -> A01:1, A11:-1
    ]
    
    # U is (4, 56)
    # U[i*2+j][col]
    # Let's just build it in the loop for clarity and correctness.
    
    for t in range(8):
        for r in range(7):
            col = 7 * t + r
            
            # U updates
            if r == 0:
                U[0][col] = 1
                U[3][col] = 1
            elif r == 1:
                U[2][col] = 1
                U[3][col] = 1
            elif r == 2:
                U[0][col] = 1
            elif r == 3:
                U[3][col] = 1
            elif r == 4:
                U[0][col] = 1
                U[1][col] = 1
            elif r == 5:
                U[2][col] = 1
                U[0][col] = -1
            elif r == 6:
                U[1][col] = 1
                U[3][col] = -1
            
            # V updates
            # B indices for block t:
            # e = B[0, 2t] -> idx 2t
            # f = B[0, 2t+1] -> idx 2t+1
            # g = B[1, 2t] -> idx 16+2t
            # h = B[1, 2t+1] -> idx 16+2t+1
            
            idx_e = 2 * t
            idx_f = 2 * t + 1
            idx_g = 16 + 2 * t
            idx_h = 16 + 2 * t + 1
            
            if r == 0:
                V[idx_e][col] = 1
                V[idx_h][col] = 1
            elif r == 1:
                V[idx_e][col] = 1
            elif r == 2:
                V[idx_f][col] = 1
                V[idx_h][col] = -1
            elif r == 3:
                V[idx_g][col] = 1
                V[idx_e][col] = -1
            elif r == 4:
                V[idx_h][col] = 1
            elif r == 5:
                V[idx_e][col] = 1
                V[idx_f][col] = 1
            elif r == 6:
                V[idx_g][col] = 1
                V[idx_h][col] = 1
            
            # W updates
            # W indices for block t:
            # C00 -> 4t
            # C10 -> 4t+1
            # C01 -> 4t+2
            # C11 -> 4t+3
            
            idx_c00 = 4 * t
            idx_c10 = 4 * t + 1
            idx_c01 = 4 * t + 2
            idx_c11 = 4 * t + 3
            
            if r == 0:
                W[idx_c00][col] = 1
                W[idx_c11][col] = 1
            elif r == 1:
                W[idx_c10][col] = 1
                W[idx_c11][col] = -1
            elif r == 2:
                W[idx_c01][col] = 1
                W[idx_c11][col] = 1
            elif r == 3:
                W[idx_c00][col] = 1
                W[idx_c10][col] = 1
            elif r == 4:
                W[idx_c00][col] = -1
                W[idx_c01][col] = 1
            elif r == 5:
                W[idx_c11][col] = 1
            elif r == 6:
                W[idx_c00][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
