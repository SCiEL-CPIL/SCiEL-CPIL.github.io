"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 40: 4 blocks, each with 5 separate Strassen + 4 outer,
    # plus 2 shared Strassen per pair of blocks (2 pairs) = 4*(5+4) + 2*2 = 36 + 4 = 40
    rank = 40
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Process blocks in pairs: (0,1) and (2,3)
    for pair in range(2):
        b0 = 2 * pair      # first block in pair
        b1 = 2 * pair + 1  # second block in pair
        
        k0 = 2 * b0    # 0 or 4
        k1 = 2 * b0 + 1 # 1 or 5
        k2 = 2 * b1    # 2 or 6
        k3 = 2 * b1 + 1 # 3 or 7
        
        # --- Shared Strassen terms (2 terms) ---
        
        # Shared M1: (A00+A11) * (B0k0+B1k1) for block b0, and (B0k2+B1k3) for block b1
        # Term 1: (A00+A11) * (B0k0 + B1k1)
        U[0 * m + 0][col] = 1  # A00
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + k0][col] = 1  # B0k0
        V[1 * p + k1][col] = 1  # B1k1
        W[k0 * n + 0][col] = 1   # C0k0 += 1
        W[k1 * n + 1][col] = 1   # C1k1 += 1
        col += 1
        
        # Term 2: (A00+A11) * (B0k2 + B1k3)
        U[0 * m + 0][col] = 1  # A00
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + k2][col] = 1  # B0k2
        V[1 * p + k3][col] = 1  # B1k3
        W[k2 * n + 0][col] = 1   # C0k2 += 1
        W[k3 * n + 1][col] = 1   # C1k3 += 1
        col += 1
        
        # Shared M4: A11 * (B1k0-B0k0) for block b0, and (B1k2-B0k2) for block b1
        # Term 3: A11 * (B1k0 - B0k0)
        U[1 * m + 1][col] = 1  # A11
        V[1 * p + k0][col] = 1   # B1k0
        V[0 * p + k0][col] = -1  # B0k0
        W[k0 * n + 0][col] = 1   # C0k0 += 1
        W[k0 * n + 1][col] = 1   # C1k0 += 1
        col += 1
        
        # Term 4: A11 * (B1k2 - B0k2)
        U[1 * m + 1][col] = 1  # A11
        V[1 * p + k2][col] = 1   # B1k2
        V[0 * p + k2][col] = -1  # B0k2
        W[k2 * n + 0][col] = 1   # C0k2 += 1
        W[k2 * n + 1][col] = 1   # C1k2 += 1
        col += 1
        
        # --- Separate Strassen terms for block b0 (5 terms) ---
        
        # M2: (A10+A11) * B0k0
        U[1 * m + 0][col] = 1  # A10
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + k0][col] = 1  # B0k0
        W[k0 * n + 1][col] = 1   # C1k0 += 1
        W[k1 * n + 1][col] = -1  # C1k1 += -1
        col += 1
        
        # M3: A00 * (B0k1 - B1k1)
        U[0 * m + 0][col] = 1  # A00
        V[0 * p + k1][col] = 1   # B0k1
        V[1 * p + k1][col] = -1  # B1k1
        W[k1 * n + 0][col] = 1   # C0k1 += 1
        W[k1 * n + 1][col] = 1   # C1k1 += 1
        col += 1
        
        # M5: (A00+A01) * B1k1
        U[0 * m + 0][col] = 1  # A00
        U[0 * m + 1][col] = 1  # A01
        V[1 * p + k1][col] = 1  # B1k1
        W[k0 * n + 0][col] = -1  # C0k0 += -1
        W[k1 * n + 0][col] = 1   # C0k1 += 1
        col += 1
        
        # M6: (A10-A00) * (B0k0+B0k1)
        U[1 * m + 0][col] = 1   # A10
        U[0 * m + 0][col] = -1  # A00
        V[0 * p + k0][col] = 1   # B0k0
        V[0 * p + k1][col] = 1   # B0k1
        W[k1 * n + 1][col] = 1   # C1k1 += 1
        col += 1
        
        # M7: (A01-A11) * (B1k0+B1k1)
        U[0 * m + 1][col] = 1   # A01
        U[1 * m + 1][col] = -1  # A11
        V[1 * p + k0][col] = 1   # B1k0
        V[1 * p + k1][col] = 1   # B1k1
        W[k0 * n + 0][col] = 1   # C0k0 += 1
        col += 1
        
        # --- Separate Strassen terms for block b1 (5 terms) ---
        
        # M2: (A10+A11) * B0k2
        U[1 * m + 0][col] = 1  # A10
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + k2][col] = 1  # B0k2
        W[k2 * n + 1][col] = 1   # C1k2 += 1
        W[k3 * n + 1][col] = -1  # C1k3 += -1
        col += 1
        
        # M3: A00 * (B0k3 - B1k3)
        U[0 * m + 0][col] = 1  # A00
        V[0 * p + k3][col] = 1   # B0k3
        V[1 * p + k3][col] = -1  # B1k3
        W[k3 * n + 0][col] = 1   # C0k3 += 1
        W[k3 * n + 1][col] = 1   # C1k3 += 1
        col += 1
        
        # M5: (A00+A01) * B1k3
        U[0 * m + 0][col] = 1  # A00
        U[0 * m + 1][col] = 1  # A01
        V[1 * p + k3][col] = 1  # B1k3
        W[k2 * n + 0][col] = -1  # C0k2 += -1
        W[k3 * n + 0][col] = 1   # C0k3 += 1
        col += 1
        
        # M6: (A10-A00) * (B0k2+B0k3)
        U[1 * m + 0][col] = 1   # A10
        U[0 * m + 0][col] = -1  # A00
        V[0 * p + k2][col] = 1   # B0k2
        V[0 * p + k3][col] = 1   # B0k3
        W[k3 * n + 1][col] = 1   # C1k3 += 1
        col += 1
        
        # M7: (A01-A11) * (B1k2+B1k3)
        U[0 * m + 1][col] = 1   # A01
        U[1 * m + 1][col] = -1  # A11
        V[1 * p + k2][col] = 1   # B1k2
        V[1 * p + k3][col] = 1   # B1k3
        W[k2 * n + 0][col] = 1   # C0k2 += 1
        col += 1
        
        # --- Outer products for block b0 (4 terms) ---
        U[0 * m + 2][col] = 1  # A02
        V[2 * p + k0][col] = 1  # B2k0
        W[k0 * n + 0][col] = 1   # C0k0 += 1
        col += 1
        
        U[0 * m + 2][col] = 1  # A02
        V[2 * p + k1][col] = 1  # B2k1
        W[k1 * n + 0][col] = 1   # C0k1 += 1
        col += 1
        
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + k0][col] = 1  # B2k0
        W[k0 * n + 1][col] = 1   # C1k0 += 1
        col += 1
        
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + k1][col] = 1  # B2k1
        W[k1 * n + 1][col] = 1   # C1k1 += 1
        col += 1
        
        # --- Outer products for block b1 (4 terms) ---
        U[0 * m + 2][col] = 1  # A02
        V[2 * p + k2][col] = 1  # B2k2
        W[k2 * n + 0][col] = 1   # C0k2 += 1
        col += 1
        
        U[0 * m + 2][col] = 1  # A02
        V[2 * p + k3][col] = 1  # B2k3
        W[k3 * n + 0][col] = 1   # C0k3 += 1
        col += 1
        
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + k2][col] = 1  # B2k2
        W[k2 * n + 1][col] = 1   # C1k2 += 1
        col += 1
        
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + k3][col] = 1  # B2k3
        W[k3 * n + 1][col] = 1   # C1k3 += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
