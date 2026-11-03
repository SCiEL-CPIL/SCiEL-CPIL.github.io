"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = (a11 + a22)(x11 + x22)
        U[0*m+0][col] = 1
        U[1*m+1][col] = 1
        V[0*p+k0][col] = 1
        V[1*p+k1][col] = 1
        W[k0*n+0][col] = 1
        W[k1*n+1][col] = 1
        col += 1
        
        # M2 = (a21 + a22) x11
        U[1*m+0][col] = 1
        U[1*m+1][col] = 1
        V[0*p+k0][col] = 1
        W[k0*n+1][col] = 1
        W[k1*n+1][col] = -1
        col += 1
        
        # M3 = a11 (x12 - x22)
        U[0*m+0][col] = 1
        V[0*p+k1][col] = 1
        V[1*p+k1][col] = -1
        W[k1*n+0][col] = 1
        W[k1*n+1][col] = 1
        col += 1
        
        # M4 = a22 (x21 - x11)
        U[1*m+1][col] = 1
        V[1*p+k0][col] = 1
        V[0*p+k0][col] = -1
        W[k0*n+0][col] = 1
        W[k0*n+1][col] = 1
        col += 1
        
        # M5 = (a11 + a12) x22
        U[0*m+0][col] = 1
        U[0*m+1][col] = 1
        V[1*p+k1][col] = 1
        W[k0*n+0][col] = -1
        W[k1*n+0][col] = 1
        col += 1
        
        # M6 = (a21 - a11) (x11 + x12)
        U[1*m+0][col] = 1
        U[0*m+0][col] = -1
        V[0*p+k0][col] = 1
        V[0*p+k1][col] = 1
        W[k1*n+1][col] = 1
        col += 1
        
        # M7 = (a12 - a22) (x21 + x22)
        U[0*m+1][col] = 1
        U[1*m+1][col] = -1
        V[1*p+k0][col] = 1
        V[1*p+k1][col] = 1
        W[k0*n+0][col] = 1
        col += 1
        
        # P1 = a13 y11
        U[0*m+2][col] = 1
        V[2*p+k0][col] = 1
        W[k0*n+0][col] = 1
        col += 1
        
        # P2 = a13 y12
        U[0*m+2][col] = 1
        V[2*p+k1][col] = 1
        W[k1*n+0][col] = 1
        col += 1
        
        # P3 = a23 y11
        U[1*m+2][col] = 1
        V[2*p+k0][col] = 1
        W[k0*n+1][col] = 1
        col += 1
        
        # P4 = a23 y12
        U[1*m+2][col] = 1
        V[2*p+k1][col] = 1
        W[k1*n+1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
