"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 6 x 50
    V = [[0] * rank for _ in range(m * p)]  # 30 x 50
    W = [[0] * rank for _ in range(p * n)]  # 20 x 50
    
    col = 0
    # 10 terms per pair of columns, 5 pairs = 50
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # For each pair of columns, use 10 multiplications
        # C0k0 = A00*B0k0 + A01*B1k0 + A02*B2k0
        # C1k0 = A10*B0k0 + A11*B1k0 + A12*B2k0
        # C0k1 = A00*B0k1 + A01*B1k1 + A02*B2k1
        # C1k1 = A10*B0k1 + A11*B1k1 + A12*B2k1
        
        # Group 1: (A00 + A01) * B0k0
        U[0][col] += 1
        U[3][col] += 1
        V[k0][col] += 1
        W[k0*2+0][col] += 1
        col += 1
        
        # Group 2: (A00 + A01) * B0k1
        U[0][col] += 1
        U[3][col] += 1
        V[k1][col] += 1
        W[k1*2+0][col] += 1
        col += 1
        
        # Group 3: (A00 + A01) * B1k0
        U[0][col] += 1
        U[3][col] += 1
        V[10+k0][col] += 1
        W[k0*2+0][col] += 1
        col += 1
        
        # Group 4: (A00 + A01) * B1k1
        U[0][col] += 1
        U[3][col] += 1
        V[10+k1][col] += 1
        W[k1*2+0][col] += 1
        col += 1
        
        # Group 5: (A10 + A11) * B0k0
        U[1][col] += 1
        U[4][col] += 1
        V[k0][col] += 1
        W[k0*2+1][col] += 1
        col += 1
        
        # Group 6: (A10 + A11) * B0k1
        U[1][col] += 1
        U[4][col] += 1
        V[k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1
        
        # Group 7: (A10 + A11) * B1k0
        U[1][col] += 1
        U[4][col] += 1
        V[10+k0][col] += 1
        W[k0*2+1][col] += 1
        col += 1
        
        # Group 8: (A10 + A11) * B1k1
        U[1][col] += 1
        U[4][col] += 1
        V[10+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1
        
        # Group 9: A02 * (B2k0 + B2k1)
        U[2][col] += 1
        V[20+k0][col] += 1
        V[20+k1][col] += 1
        W[k0*2+0][col] += 1
        W[k1*2+0][col] += 1
        col += 1
        
        # Group 10: A12 * (B2k0 + B2k1)
        U[5][col] += 1
        V[20+k0][col] += 1
        V[20+k1][col] += 1
        W[k0*2+1][col] += 1
        W[k1*2+1][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
