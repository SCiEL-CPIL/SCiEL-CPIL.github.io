"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 Strassen-like decomposition
    # 5 blocks of 2x2
    # Each block uses 7 products
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B22-B12)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U coefficients (for A)
    # Rows of U correspond to A11, A12, A21, A22
    # Col 0: M1 -> A11:1, A22:1
    # Col 1: M2 -> A21:1, A22:1
    # Col 2: M3 -> A11:1
    # Col 3: M4 -> A22:1
    # Col 4: M5 -> A11:1, A12:1
    # Col 5: M6 -> A21:1, A11:-1
    # Col 6: M7 -> A12:1, A22:-1
    
    u_coeffs = [
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ]
    
    # V coefficients (for B)
    # Rows of V correspond to B11, B12, B21, B22 (for a 2x2 block)
    # Col 0: M1 -> B11:1, B22:1
    # Col 1: M2 -> B11:1
    # Col 2: M3 -> B22:1, B12:-1
    # Col 3: M4 -> B11:1, B21:-1
    # Col 4: M5 -> B22:1
    # Col 5: M6 -> B11:1, B12:1
    # Col 6: M7 -> B21:1, B22:1
    
    v_coeffs = [
        [1, 1, 0, 1, 0, 1, 0], # B11
        [0, 0, -1, 0, 0, 1, 0],# B12
        [0, 0, 0, -1, 0, 0, 1],# B21
        [1, 0, 1, 0, 1, 0, 1]  # B22
    ]
    
    # W coefficients (for C)
    # Rows of W correspond to C11, C12, C21, C22
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M4 + M7
    # C21 = M2 + M3 - M1 + M5
    # C22 = M1 - M2 + M5 - M6
    
    w_coeffs = [
        [1, 0, 0, -1, 0, 1, -1], # C11
        [0, 0, 1, 1, 0, 0, 1],   # C12
        [-1, 1, 1, 0, 1, 0, 0],  # C21
        [1, -1, 0, 0, 1, -1, 0]  # C22
    ]
    
    # Map to global indices
    # A is 2x2, indices 0..3
    # B is 2x10, indices 0..19
    # C is 2x10, indices 0..19
    
    # B indices for block b (0..4)
    # B11: b*4 + 0
    # B12: b*4 + 1
    # B21: b*4 + 2
    # B22: b*4 + 3
    
    # C indices for block b (0..4)
    # C11: b*2 + 0
    # C12: b*2 + 1
    # C21: b*2 + 2
    # C22: b*2 + 3
    
    for b in range(5):
        for k in range(7):
            col = b * 7 + k
            # Fill U
            for i in range(4):
                U[i][col] = u_coeffs[i][k]
            
            # Fill V
            # v_coeffs[i] corresponds to B11, B12, B21, B22
            # Global B index:
            # B11: b*4 + 0
            # B12: b*4 + 1
            # B21: b*4 + 2
            # B22: b*4 + 3
            for i in range(4):
                row = b * 4 + i
                V[row][col] = v_coeffs[i][k]
            
            # Fill W
            # w_coeffs[i] corresponds to C11, C12, C21, C22
            # Global C index:
            # C11: b*2 + 0
            # C12: b*2 + 1
            # C21: b*2 + 2
            # C22: b*2 + 3
            for i in range(4):
                row = b * 2 + i
                W[row][col] = w_coeffs[i][k]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
