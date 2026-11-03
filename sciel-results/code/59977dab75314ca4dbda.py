"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank-17 decomposition of 2x3x3 matrix multiplication.
    # Structure:
    # 1. j=2 part (6 terms): A[i,2] * B[2,k]
    # 2. j in {0,1} part (11 terms):
    #    - Strassen for 2x2x2 (7 terms)
    #    - Schoolbook for 2x2x1 (4 terms)
    
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(mat, row, col, val):
        mat[row][col] += val

    # Part 1: j=2 (index 2)
    # O[i,k] = A[i,2] * B[2,k]
    # U index: i*3 + 2
    # V index: 2*3 + k
    # W index: k*2 + i
    
    col = 0
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            set_val(U, u_idx, col, 1)
            set_val(V, v_idx, col, 1)
            set_val(W, w_idx, col, 1)
            col += 1
            
    # Part 2: j in {0, 1}
    # A' is 2x2, B' is 2x3.
    # B' split into B'_1 (2x2, cols 0,1) and B'_2 (2x1, col 2).
    
    # Strassen for A' * B'_1 (2x2 * 2x2)
    # M1 = (A00 + A11) * (B00 + B11)
    set_val(U, 0, col, 1)   # A00
    set_val(U, 4, col, 1)   # A11
    set_val(V, 0, col, 1)   # B00
    set_val(V, 4, col, 1)   # B11
    set_val(W, 0, col, 1)   # C00
    set_val(W, 3, col, 1)   # C11
    col += 1
    
    # M2 = (A10 + A11) * B00
    set_val(U, 3, col, 1)   # A10
    set_val(U, 4, col, 1)   # A11
    set_val(V, 0, col, 1)   # B00
    set_val(W, 1, col, 1)   # C10
    set_val(W, 3, col, -1)  # C11
    col += 1
    
    # M3 = A00 * (B01 - B11)
    set_val(U, 0, col, 1)   # A00
    set_val(V, 1, col, 1)   # B01
    set_val(V, 4, col, -1)  # B11
    set_val(W, 2, col, 1)   # C01
    set_val(W, 3, col, 1)   # C11
    col += 1
    
    # M4 = A11 * (B10 - B00)
    set_val(U, 4, col, 1)   # A11
    set_val(V, 3, col, 1)   # B10
    set_val(V, 0, col, -1)  # B00
    set_val(W, 0, col, 1)   # C00
    set_val(W, 1, col, 1)   # C10
    col += 1
    
    # M5 = (A00 + A01) * B11
    set_val(U, 0, col, 1)   # A00
    set_val(U, 1, col, 1)   # A01
    set_val(V, 4, col, 1)   # B11
    set_val(W, 0, col, -1)  # C00
    set_val(W, 2, col, 1)   # C01
    col += 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    set_val(U, 3, col, 1)   # A10
    set_val(U, 0, col, -1)  # A00
    set_val(V, 0, col, 1)   # B00
    set_val(V, 1, col, 1)   # B01
    set_val(W, 3, col, 1)   # C11
    col += 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    set_val(U, 1, col, 1)   # A01
    set_val(U, 4, col, -1)  # A11
    set_val(V, 3, col, 1)   # B10
    set_val(V, 4, col, 1)   # B11
    set_val(W, 0, col, 1)   # C00
    col += 1
    
    # Schoolbook for A' * B'_2 (2x2 * 2x1)
    # N1 = A00 * B02
    set_val(U, 0, col, 1)   # A00
    set_val(V, 2, col, 1)   # B02
    set_val(W, 4, col, 1)   # C02
    col += 1
    
    # N2 = A01 * B12
    set_val(U, 1, col, 1)   # A01
    set_val(V, 5, col, 1)   # B12
    set_val(W, 4, col, 1)   # C02
    col += 1
    
    # N3 = A10 * B02
    set_val(U, 3, col, 1)   # A10
    set_val(V, 2, col, 1)   # B02
    set_val(W, 5, col, 1)   # C12
    col += 1
    
    # N4 = A11 * B12
    set_val(U, 4, col, 1)   # A11
    set_val(V, 5, col, 1)   # B12
    set_val(W, 5, col, 1)   # C12
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
