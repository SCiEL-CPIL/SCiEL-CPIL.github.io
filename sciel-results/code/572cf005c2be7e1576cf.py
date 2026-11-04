"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add to matrices
    def add_U(i, j, col, val):
        U[i * m + j][col] += val
        
    def add_V(j, k, col, val):
        V[j * p + k][col] += val
        
    def add_W(k, i, col, val):
        W[k * n + i][col] += val

    # Loop over the 3 groups of k (columns of B, columns of C)
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        s = t * 11  # Start column index for this group
        
        # --- Term 1: 2x2x2 Strassen for j in {0,1}, k in {k0, k1} ---
        
        # M1 = (A00 + A11) * (B00 + B11)
        add_U(0, 0, s, 1)
        add_U(1, 1, s, 1)
        add_V(0, k0, s, 1)
        add_V(1, k1, s, 1)
        add_W(k0, 0, s, 1)
        add_W(k1, 1, s, 1)
        
        # M2 = (A10 + A11) * B00
        add_U(1, 0, s+1, 1)
        add_U(1, 1, s+1, 1)
        add_V(0, k0, s+1, 1)
        add_W(k0, 1, s+1, 1)
        add_W(k1, 1, s+1, -1)
        
        # M3 = A00 * (B10 - B11)
        add_U(0, 0, s+2, 1)
        add_V(1, k0, s+2, 1)
        add_V(1, k1, s+2, -1)
        add_W(k1, 0, s+2, 1)
        add_W(k1, 1, s+2, 1)
        
        # M4 = A11 * (B01 - B00)
        add_U(1, 1, s+3, 1)
        add_V(0, k1, s+3, 1)
        add_V(0, k0, s+3, -1)
        add_W(k0, 0, s+3, 1)
        add_W(k0, 1, s+3, 1)
        
        # M5 = (A00 + A01) * B11
        add_U(0, 0, s+4, 1)
        add_U(0, 1, s+4, 1)
        add_V(1, k1, s+4, 1)
        add_W(k0, 0, s+4, -1)
        add_W(k1, 0, s+4, 1)
        
        # M6 = (A10 - A00) * (B00 + B01)
        add_U(1, 0, s+5, 1)
        add_U(0, 0, s+5, -1)
        add_V(0, k0, s+5, 1)
        add_V(0, k1, s+5, 1)
        add_W(k1, 1, s+5, 1)
        
        # M7 = (A01 - A11) * (B10 + B11)
        add_U(0, 1, s+6, 1)
        add_U(1, 1, s+6, -1)
        add_V(1, k0, s+6, 1)
        add_V(1, k1, s+6, 1)
        add_W(k0, 0, s+6, 1)
        add_W(k1, 0, s+6, -1)
        
        # --- Term 2: 2x1x2 Outer Product for j=2, k in {k0, k1} ---
        
        # A02 * B2,k0
        add_U(0, 2, s+7, 1)
        add_V(2, k0, s+7, 1)
        add_W(k0, 0, s+7, 1)
        
        # A02 * B2,k1
        add_U(0, 2, s+8, 1)
        add_V(2, k1, s+8, 1)
        add_W(k1, 0, s+8, 1)
        
        # A12 * B2,k0
        add_U(1, 2, s+9, 1)
        add_V(2, k0, s+9, 1)
        add_W(k0, 1, s+9, 1)
        
        # A12 * B2,k1
        add_U(1, 2, s+10, 1)
        add_V(2, k1, s+10, 1)
        add_W(k1, 1, s+10, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
