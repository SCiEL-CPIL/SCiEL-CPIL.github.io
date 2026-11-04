"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition:
    # 1. M_{2,1,10} part (rank 20)
    # 2. M_{2,2,10} part split into 5 M_{2,2,2} Strassen blocks (rank 35)
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: M_{2,1,10}
    # A uses column 0 (j=0). B uses row 0 (j=0).
    # C is 2x10.
    col = 0
    for i in range(n):
        for k in range(p):
            # U index: i*m + 0
            U[i * m + 0][col] = 1
            # V index: 0*p + k
            V[0 * p + k][col] = 1
            # W index: k*n + i
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: M_{2,2,10}
    # A uses columns 1, 2 (j=1, 2). B uses rows 1, 2 (j=1, 2).
    # Split B columns into 5 pairs: (0,1), (2,3), ..., (8,9)
    # Each pair is a 2x2 multiplication with A (2x2).
    
    # Strassen factors for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6
    
    # Local A indices:
    # A00 -> global (0, 1)
    # A01 -> global (0, 2)
    # A10 -> global (1, 1)
    # A11 -> global (1, 2)
    
    # Local B indices for block b (cols 2b, 2b+1):
    # B00 -> global (1, 2b)
    # B01 -> global (1, 2b+1)
    # B10 -> global (2, 2b)
    # B11 -> global (2, 2b+1)
    
    # Local C indices for block b:
    # C00 -> global (0, 2b)
    # C01 -> global (0, 2b+1)
    # C10 -> global (1, 2b)
    # C11 -> global (1, 2b+1)
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1
        c = col
        U[0 * m + 1][c] = 1
        U[1 * m + 2][c] = 1
        V[1 * p + k0][c] = 1
        V[2 * p + k1][c] = 1
        W[k0 * n + 0][c] = 1
        W[k1 * n + 1][c] = 1
        col += 1
        
        # M2
        c = col
        U[1 * m + 1][c] = 1
        U[1 * m + 2][c] = 1
        V[1 * p + k0][c] = 1
        W[k0 * n + 1][c] = 1
        col += 1
        
        # M3
        c = col
        U[0 * m + 1][c] = 1
        V[1 * p + k1][c] = 1
        V[2 * p + k1][c] = -1
        W[k1 * n + 0][c] = 1
        W[k1 * n + 1][c] = -1
        col += 1
        
        # M4
        c = col
        U[1 * m + 2][c] = 1
        V[2 * p + k0][c] = 1
        V[1 * p + k0][c] = -1
        W[k0 * n + 0][c] = -1
        W[k0 * n + 1][c] = 1
        col += 1
        
        # M5
        c = col
        U[0 * m + 1][c] = 1
        U[0 * m + 2][c] = 1
        V[2 * p + k1][c] = 1
        W[k0 * n + 0][c] = 1
        col += 1
        
        # M6
        c = col
        U[1 * m + 1][c] = 1
        U[0 * m + 1][c] = -1
        V[1 * p + k0][c] = 1
        V[1 * p + k1][c] = 1
        W[k1 * n + 1][c] = 1
        col += 1
        
        # M7
        c = col
        U[0 * m + 2][c] = 1
        U[1 * m + 2][c] = -1
        V[2 * p + k0][c] = 1
        V[2 * p + k1][c] = 1
        W[k0 * n + 0][c] = -1
        W[k1 * n + 0][c] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
