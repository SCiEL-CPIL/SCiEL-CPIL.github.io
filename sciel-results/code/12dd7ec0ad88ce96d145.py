"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 10
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # U indices: 0:a, 1:b, 2:c, 3:d
    # V indices for block t: 
    #   e: 2t
    #   f: 2t+1
    #   g: 10+2t
    #   h: 10+2t+1
    # W indices for block t:
    #   C00: 4t
    #   C10: 4t+1
    #   C01: 4t+2
    #   C11: 4t+3
    
    for t in range(5):
        base_r = 7 * t
        # M1
        r = base_r
        U[0][r] = 1; U[3][r] = 1
        V[2*t][r] = 1; V[10+2*t+1][r] = 1
        W[4*t][r] = 1; W[4*t+3][r] = 1
        
        # M2
        r = base_r + 1
        U[2][r] = 1; U[3][r] = 1
        V[2*t][r] = 1
        W[4*t+1][r] = 1; W[4*t+3][r] = -1
        
        # M3
        r = base_r + 2
        U[0][r] = 1
        V[2*t+1][r] = 1; V[10+2*t+1][r] = -1
        W[4*t+2][r] = 1; W[4*t+3][r] = 1
        
        # M4
        r = base_r + 3
        U[3][r] = 1
        V[10+2*t][r] = 1; V[2*t][r] = -1
        W[4*t][r] = 1; W[4*t+1][r] = 1
        
        # M5
        r = base_r + 4
        U[0][r] = 1; U[1][r] = 1
        V[10+2*t+1][r] = 1
        W[4*t][r] = -1; W[4*t+2][r] = 1
        
        # M6
        r = base_r + 5
        U[2][r] = 1; U[0][r] = -1
        V[2*t][r] = 1; V[2*t+1][r] = 1
        W[4*t+3][r] = 1
        
        # M7
        r = base_r + 6
        U[1][r] = 1; U[3][r] = -1
        V[10+2*t][r] = 1; V[10+2*t+1][r] = 1
        W[4*t][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
