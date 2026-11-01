"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank = 6 * 7 + 4 = 46
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 6 blocks of 2x2x2
    for t in range(6):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Term 1: (A00 + A11)(B0k0 + B1k1)
        # U: A00=1, A11=1
        U[0][col] = 1
        U[3][col] = 1
        # V: B0k0=1, B1k1=1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        # W: C0k0+=1, C1k1+=1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # Term 2: (A10 + A11)B0k0
        U[2][col] = 1 # A10 (i=1, j=0) -> index 1*2+0=2
        U[3][col] = 1 # A11
        V[0 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1 # C1k0
        W[k1 * n + 1][col] = -1 # C1k1 (from -m2)
        col += 1
        
        # Term 3: A00(B1k0 - B0k0)
        U[0][col] = 1 # A00
        V[1 * p + k0][col] = 1 # B1k0
        V[0 * p + k0][col] = -1 # B0k0
        W[k1 * n + 0][col] = 1 # C0k1
        W[k1 * n + 1][col] = 1 # C1k1 (from +m3)
        col += 1
        
        # Term 4: A11(B0k1 - B1k1)
        U[3][col] = 1 # A11
        V[0 * p + k1][col] = 1 # B0k1
        V[1 * p + k1][col] = -1 # B1k1
        W[k0 * n + 0][col] = 1 # C0k0
        W[k0 * n + 1][col] = 1 # C1k0
        col += 1
        
        # Term 5: (A00 + A01)B1k1
        U[0][col] = 1 # A00
        U[1][col] = 1 # A01
        V[1 * p + k1][col] = 1 # B1k1
        W[k0 * n + 0][col] = -1 # C0k0
        W[k1 * n + 0][col] = 1 # C0k1
        col += 1
        
        # Term 6: (A10 - A00)(B0k0 + B0k1)
        U[2][col] = 1 # A10
        U[0][col] = -1 # A00
        V[0 * p + k0][col] = 1 # B0k0
        V[0 * p + k1][col] = 1 # B0k1
        W[k1 * n + 1][col] = 1 # C1k1
        col += 1
        
        # Term 7: (A01 - A11)(B1k0 + B1k1)
        U[1][col] = 1 # A01
        U[3][col] = -1 # A11
        V[1 * p + k0][col] = 1 # B1k0
        V[1 * p + k1][col] = 1 # B1k1
        W[k0 * n + 0][col] = 1 # C0k0
        col += 1
        
    # 1 block of 2x2x1 (col 12)
    k0 = 12
    
    # Term 1: A00 B0k0 -> C0k0
    U[0][col] = 1
    V[0 * p + k0][col] = 1
    W[k0 * n + 0][col] = 1
    col += 1
    
    # Term 2: A01 B1k0 -> C0k0
    U[1][col] = 1
    V[1 * p + k0][col] = 1
    W[k0 * n + 0][col] = 1
    col += 1
    
    # Term 3: A10 B0k0 -> C1k0
    U[2][col] = 1
    V[0 * p + k0][col] = 1
    W[k0 * n + 1][col] = 1
    col += 1
    
    # Term 4: A11 B1k0 -> C1k0
    U[3][col] = 1
    V[1 * p + k0][col] = 1
    W[k0 * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
