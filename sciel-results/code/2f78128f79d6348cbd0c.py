"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 25
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for B columns k0, k0+1
    def add_strassen_block(k0):
        nonlocal col
        # M1: (A00+A11)(B00+B11)
        U[0][col] = 1; U[3][col] = 1
        V[k0][col] = 1; V[k0+1][col] = 1
        W[k0*2 + 0][col] = 1; W[(k0+1)*2 + 1][col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        U[2][col] = 1; U[3][col] = 1
        V[k0][col] = 1
        W[k0*2 + 1][col] = 1; W[(k0+1)*2 + 1][col] = -1
        col += 1
        
        # M3: A00(B01-B11)
        U[0][col] = 1
        V[k0+1][col] = 1; V[k0+1+1][col] = -1 # Note: B indices are j*p+k.
        # Wait, V index is j*p + k.
        # Local B00 is j=0, k=k0. Index 0*7+k0 = k0.
        # Local B01 is j=0, k=k0+1. Index 0*7+k0+1 = k0+1.
        # Local B10 is j=1, k=k0. Index 1*7+k0 = 7+k0.
        # Local B11 is j=1, k=k0+1. Index 1*7+k0+1 = 7+k0+1.
        
        # Correcting V indices for Strassen:
        # M1: B00 (j=0,k=k0), B11 (j=1,k=k0+1)
        # M2: B00 (j=0,k=k0)
        # M3: B01 (j=0,k=k0+1), B11 (j=1,k=k0+1)
        # M4: B10 (j=1,k=k0), B00 (j=0,k=k0)
        # M5: B11 (j=1,k=k0+1)
        # M6: B00 (j=0,k=k0), B01 (j=0,k=k0+1)
        # M7: B10 (j=1,k=k0), B11 (j=1,k=k0+1)
        
        # Let's re-implement carefully.
        
    # Reset col
    col = 0
    
    # Strassen Block 1: k=0,1
    # M1
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[7+1][col] = 1 # B00, B11
    W[0*2+0][col] = 1; W[1*2+1][col] = 1 # C00, C11
    col += 1
    # M2
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1 # B00
    W[0*2+1][col] = 1; W[1*2+1][col] = -1 # C10, C11
    col += 1
    # M3
    U[0][col] = 1
    V[1][col] = 1; V[7+1][col] = -1 # B01, B11
    W[1*2+0][col] = 1; W[1*2+1][col] = 1 # C01, C11
    col += 1
    # M4
    U[3][col] = 1
    V[7+0][col] = 1; V[0][col] = -1 # B10, B00
    W[0*2+0][col] = 1; W[0*2+1][col] = 1 # C00, C10
    col += 1
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[7+1][col] = 1 # B11
    W[0*2+0][col] = -1; W[1*2+0][col] = 1 # C00, C01
    col += 1
    # M6
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1 # B00, B01
    W[1*2+1][col] = 1 # C11
    col += 1
    # M7
    U[1][col] = 1; U[3][col] = -1
    V[7+0][col] = 1; V[7+1][col] = 1 # B10, B11
    W[0*2+0][col] = 1 # C00
    col += 1
    
    # Strassen Block 2: k=2,3
    # M1
    U[0][col] = 1; U[3][col] = 1
    V[2][col] = 1; V[7+3][col] = 1
    W[2*2+0][col] = 1; W[3*2+1][col] = 1
    col += 1
    # M2
    U[2][col] = 1; U[3][col] = 1
    V[2][col] = 1
    W[2*2+1][col] = 1; W[3*2+1][col] = -1
    col += 1
    # M3
    U[0][col] = 1
    V[3][col] = 1; V[7+3][col] = -1
    W[3*2+0][col] = 1; W[3*2+1][col] = 1
    col += 1
    # M4
    U[3][col] = 1
    V[7+2][col] = 1; V[2][col] = -1
    W[2*2+0][col] = 1; W[2*2+1][col] = 1
    col += 1
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[7+3][col] = 1
    W[2*2+0][col] = -1; W[3*2+0][col] = 1
    col += 1
    # M6
    U[2][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[3*2+1][col] = 1
    col += 1
    # M7
    U[1][col] = 1; U[3][col] = -1
    V[7+2][col] = 1; V[7+3][col] = 1
    W[2*2+0][col] = 1
    col += 1
    
    # Strassen Block 3: k=4,5
    # M1
    U[0][col] = 1; U[3][col] = 1
    V[4][col] = 1; V[7+5][col] = 1
    W[4*2+0][col] = 1; W[5*2+1][col] = 1
    col += 1
    # M2
    U[2][col] = 1; U[3][col] = 1
    V[4][col] = 1
    W[4*2+1][col] = 1; W[5*2+1][col] = -1
    col += 1
    # M3
    U[0][col] = 1
    V[5][col] = 1; V[7+5][col] = -1
    W[5*2+0][col] = 1; W[5*2+1][col] = 1
    col += 1
    # M4
    U[3][col] = 1
    V[7+4][col] = 1; V[4][col] = -1
    W[4*2+0][col] = 1; W[4*2+1][col] = 1
    col += 1
    # M5
    U[0][col] = 1; U[1][col] = 1
    V[7+5][col] = 1
    W[4*2+0][col] = -1; W[5*2+0][col] = 1
    col += 1
    # M6
    U[2][col] = 1; U[0][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[5*2+1][col] = 1
    col += 1
    # M7
    U[1][col] = 1; U[3][col] = -1
    V[7+4][col] = 1; V[7+5][col] = 1
    W[4*2+0][col] = 1
    col += 1
    
    # Schoolbook Block 4: k=6
    # S1: A00 B0,6
    U[0][col] = 1
    V[6][col] = 1
    W[6*2+0][col] = 1
    col += 1
    # S2: A01 B1,6
    U[1][col] = 1
    V[7+6][col] = 1
    W[6*2+0][col] = 1
    col += 1
    # S3: A10 B0,6
    U[2][col] = 1
    V[6][col] = 1
    W[6*2+1][col] = 1
    col += 1
    # S4: A11 B1,6
    U[3][col] = 1
    V[7+6][col] = 1
    W[6*2+1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
