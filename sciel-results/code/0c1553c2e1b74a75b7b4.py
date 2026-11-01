"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # U: (n*m, R) = (6, R)
    # V: (m*p, R) = (30, R)
    # W: (p*n, R) = (20, R)
    
    # Rank 55
    R = 55
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    col = 0
    
    for q in range(5):
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # M1
        U[0*3+0][col] += 1.0
        U[1*3+1][col] += 1.0
        V[0*10+k0][col] += 1.0
        V[1*10+k1][col] += 1.0
        W[k0*2+0][col] += 1.0
        W[k1*2+1][col] += 1.0
        col += 1
        
        # M2
        U[1*3+0][col] += 1.0
        U[1*3+1][col] += 1.0
        V[0*10+k0][col] += 1.0
        W[k0*2+1][col] += 1.0
        col += 1
        
        # M3
        U[0*3+0][col] += 1.0
        V[1*10+k0][col] += -1.0
        V[1*10+k1][col] += 1.0
        W[k0*2+0][col] += 1.0
        W[k1*2+0][col] += 1.0
        col += 1
        
        # M4
        U[1*3+1][col] += 1.0
        V[0*10+k0][col] += -1.0
        V[0*10+k1][col] += 1.0
        W[k0*2+0][col] += 1.0
        W[k1*2+1][col] += 1.0
        col += 1
        
        # M5
        U[0*3+0][col] += 1.0
        U[0*3+1][col] += 1.0
        V[1*10+k1][col] += 1.0
        W[k0*2+0][col] += -1.0
        W[k1*2+0][col] += 1.0
        col += 1
        
        # M6
        U[1*3+0][col] += 1.0
        U[0*3+0][col] += -1.0
        V[0*10+k0][col] += 1.0
        V[0*10+k1][col] += 1.0
        W[k1*2+1][col] += 1.0
        col += 1
        
        # M7
        U[0*3+1][col] += 1.0
        U[1*3+1][col] += -1.0
        V[1*10+k0][col] += 1.0
        V[1*10+k1][col] += 1.0
        W[k0*2+0][col] += 1.0
        W[k1*2+1][col] += 1.0
        col += 1
        
        # OP00
        U[0*3+2][col] += 1.0
        V[2*10+k0][col] += 1.0
        W[k0*2+0][col] += 1.0
        col += 1
        
        # OP01
        U[0*3+2][col] += 1.0
        V[2*10+k1][col] += 1.0
        W[k1*2+0][col] += 1.0
        col += 1
        
        # OP10
        U[1*3+2][col] += 1.0
        V[2*10+k0][col] += 1.0
        W[k0*2+1][col] += 1.0
        col += 1
        
        # OP11
        U[1*3+2][col] += 1.0
        V[2*10+k1][col] += 1.0
        W[k1*2+1][col] += 1.0
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
