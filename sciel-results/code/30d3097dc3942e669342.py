"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 decomposition: 4 Strassen (2x2x2) + 1 Schoolbook (2x2x1)
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block for columns k0, k1
    def add_strassen(k0, k1):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[0][col] += 1 # A00
        U[3][col] += 1 # A11 (index 1*2+1)
        V[0*9 + k0][col] += 1 # B00
        V[1*9 + k1][col] += 1 # B11
        W[k0*2 + 0][col] += 1 # C00
        W[k0*2 + 1][col] += 1 # C11 (Wait, C11 is i=1, k=k1? No, C11 is i=1, k=k1. Index k1*2+1)
        # Correction on W indices:
        # C00 -> i=0, k=k0 -> W[k0*2+0]
        # C01 -> i=0, k=k1 -> W[k1*2+0]
        # C10 -> i=1, k=k0 -> W[k0*2+1]
        # C11 -> i=1, k=k1 -> W[k1*2+1]
        
        # Let's re-evaluate Strassen output mapping
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1
        U[0][col] += 1
        U[3][col] += 1
        V[0*9 + k0][col] += 1
        V[1*9 + k1][col] += 1
        W[k0*2 + 0][col] += 1 # C00
        W[k1*2 + 1][col] += 1 # C11
        col += 1
        
        # M2 = (A10 + A11) B00
        U[2][col] += 1 # A10 (index 1*2+0)
        U[3][col] += 1 # A11
        V[0*9 + k0][col] += 1 # B00
        W[k0*2 + 1][col] += 1 # C10
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[0][col] += 1
        V[0*9 + k1][col] += 1 # B01
        V[1*9 + k1][col] -= 1 # B11
        W[k1*2 + 0][col] += 1 # C01
        W[k1*2 + 1][col] += 1 # C11
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[3][col] += 1
        V[1*9 + k0][col] += 1 # B10
        V[0*9 + k0][col] -= 1 # B00
        W[k0*2 + 0][col] += 1 # C00
        W[k0*2 + 1][col] += 1 # C10
        col += 1
        
        # M5 = (A00 + A01) B11
        U[0][col] += 1
        U[1][col] += 1 # A01 (index 0*2+1)
        V[1*9 + k1][col] += 1 # B11
        W[k0*2 + 0][col] -= 1 # C00
        W[k1*2 + 0][col] += 1 # C01
        col += 1
        
        # M6 = (A10 - A00) (B00 + B01)
        U[2][col] += 1
        U[0][col] -= 1
        V[0*9 + k0][col] += 1
        V[0*9 + k1][col] += 1
        W[k1*2 + 1][col] += 1 # C11
        col += 1
        
        # M7 = (A01 - A11) (B10 + B11)
        U[1][col] += 1
        U[3][col] -= 1
        V[1*9 + k0][col] += 1
        V[1*9 + k1][col] += 1
        W[k0*2 + 0][col] += 1 # C00
        col += 1

    # Helper to add Schoolbook block for column k0
    def add_schoolbook(k0):
        nonlocal col
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        
        # Term 1: A00 B00
        U[0][col] += 1
        V[0*9 + k0][col] += 1
        W[k0*2 + 0][col] += 1
        col += 1
        
        # Term 2: A01 B10
        U[1][col] += 1
        V[1*9 + k0][col] += 1
        W[k0*2 + 0][col] += 1
        col += 1
        
        # Term 3: A10 B00
        U[2][col] += 1
        V[0*9 + k0][col] += 1
        W[k0*2 + 1][col] += 1
        col += 1
        
        # Term 4: A11 B10
        U[3][col] += 1
        V[1*9 + k0][col] += 1
        W[k0*2 + 1][col] += 1
        col += 1

    # 4 Strassen blocks for cols (0,1), (2,3), (4,5), (6,7)
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    
    # 1 Schoolbook block for col 8
    add_schoolbook(8)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
