"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank = 7 blocks * 7 strassen terms = 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[0][col] += 1 # A00
        U[3][col] += 1 # A11
        V[k0][col] += 1 # B00 (row 0, col k0)
        V[14 + k1][col] += 1 # B11 (row 1, col k1)
        W[k0 * 2 + 0][col] += 1 # C00
        W[k1 * 2 + 1][col] += 1 # C11
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[2][col] += 1 # A10
        U[3][col] += 1 # A11
        V[k0][col] += 1 # B00
        W[k0 * 2 + 1][col] += 1 # C10
        W[k1 * 2 + 1][col] -= 1 # C11
        col += 1
        
        # M3 = A00 * (B10 - B11)
        U[0][col] += 1 # A00
        V[14 + k0][col] += 1 # B10
        V[14 + k1][col] -= 1 # B11
        W[k1 * 2 + 0][col] += 1 # C01
        W[k1 * 2 + 1][col] += 1 # C11
        col += 1
        
        # M4 = A11 * (B01 - B00)
        U[3][col] += 1 # A11
        V[k1][col] += 1 # B01
        V[k0][col] -= 1 # B00
        W[k0 * 2 + 0][col] += 1 # C00
        W[k0 * 2 + 1][col] += 1 # C10
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[0][col] += 1 # A00
        U[1][col] += 1 # A01
        V[14 + k1][col] += 1 # B11
        W[k0 * 2 + 0][col] -= 1 # C00
        W[k1 * 2 + 0][col] += 1 # C01
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[2][col] += 1 # A10
        U[0][col] -= 1 # A00
        V[k0][col] += 1 # B00
        V[k1][col] += 1 # B01
        W[k1 * 2 + 1][col] += 1 # C11
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[1][col] += 1 # A01
        U[3][col] -= 1 # A11
        V[14 + k0][col] += 1 # B10
        V[14 + k1][col] += 1 # B11
        W[k0 * 2 + 0][col] += 1 # C00
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
