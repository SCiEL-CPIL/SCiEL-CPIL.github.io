"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 using Strassen's algorithm for 2x2 blocks
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 matmul
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        col_base = 7 * b
        
        # M1
        c = col_base
        U[0][c] = 1
        U[3][c] = 1
        V[k0][c] = 1
        V[10 + k1][c] = 1
        W[2 * k0][c] = 1
        W[2 * k1 + 1][c] = 1
        
        # M2
        c = col_base + 1
        U[2][c] = 1
        U[3][c] = 1
        V[k0][c] = 1
        W[2 * k0 + 1][c] = 1
        W[2 * k1 + 1][c] = -1
        
        # M3
        c = col_base + 2
        U[0][c] = 1
        V[k1][c] = 1
        V[10 + k1][c] = -1
        W[2 * k1][c] = 1
        W[2 * k1 + 1][c] = 1
        
        # M4
        c = col_base + 3
        U[3][c] = 1
        V[10 + k0][c] = 1
        V[k0][c] = -1
        W[2 * k0][c] = 1
        W[2 * k0 + 1][c] = 1
        
        # M5
        c = col_base + 4
        U[0][c] = 1
        U[1][c] = 1
        V[10 + k1][c] = 1
        W[2 * k0][c] = -1
        W[2 * k1][c] = 1
        
        # M6
        c = col_base + 5
        U[2][c] = 1
        U[0][c] = -1
        V[k0][c] = 1
        V[k1][c] = 1
        W[2 * k1 + 1][c] = 1
        
        # M7
        c = col_base + 6
        U[1][c] = 1
        U[3][c] = -1
        V[10 + k0][c] = 1
        V[10 + k1][c] = 1
        W[2 * k0][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
