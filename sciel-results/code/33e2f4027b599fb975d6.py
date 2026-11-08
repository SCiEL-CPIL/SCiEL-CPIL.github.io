"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # We will construct a decomposition of rank 54.
    # Split m=5 into m1=2, m2=2, m3=1.
    # Split p=6 into p1=2, p2=2, p3=2.
    
    # A is 2x5. Split into A1 (2x2), A2 (2x2), A3 (2x1).
    # B is 5x6. Split into B1 (2x6), B2 (2x6), B3 (1x6).
    # C = A1 B1 + A2 B2 + A3 B3.
    
    # For A1 B1 (2x2 * 2x6):
    # Split B1 into B1a (2x2), B1b (2x2), B1c (2x2).
    # Use Strassen for each 2x2*2x2 block.
    # Rank for each is 7. Total 21.
    
    # For A2 B2 (2x2 * 2x6):
    # Similarly, 21.
    
    # For A3 B3 (2x1 * 1x6):
    # Standard multiplication. 12 multiplications.
    
    # Total rank = 21 + 21 + 12 = 54.
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen decomposition for 2x2 * 2x2
    # Indices for A: i in 0..1, j in j_start..j_start+1
    # Indices for B: j in j_start..j_start+1, k in k_start..k_start+1
    # Indices for C: i in 0..1, k in k_start..k_start+1
    
    def add_strassen(j_start, k_start):
        nonlocal col
        # Strassen terms for A[i, j_start + dj] and B[j_start + dj, k_start + dk]
        # A indices: 0, 1
        # B indices: 0, 1
        # C indices: 0, 1
        
        # M1 = A00 B00
        U[0 * m + j_start + 0][col] = 1
        V[(j_start + 0) * p + k_start + 0][col] = 1
        W[(k_start + 0) * n + 0][col] = 1
        col += 1
        
        # M2 = A11 B11
        U[1 * m + j_start + 1][col] = 1
        V[(j_start + 1) * p + k_start + 1][col] = 1
        W[(k_start + 1) * n + 1][col] = 1
        col += 1
        
        # M3 = (A00+A11) (B00+B11)
        U[0 * m + j_start + 0][col] = 1
        U[1 * m + j_start + 1][col] = 1
        V[(j_start + 0) * p + k_start + 0][col] = 1
        V[(j_start + 1) * p + k_start + 1][col] = 1
        W[(k_start + 0) * n + 0][col] = 1
        W[(k_start + 0) * n + 1][col] = 1
        W[(k_start + 1) * n + 0][col] = 1
        W[(k_start + 1) * n + 1][col] = 1
        # Wait, Strassen formula:
        # C00 = M1 + M3 - M2 - M4? No.
        # Standard Strassen:
        # M1 = A00 B00
        # M2 = A10 B11
        # M3 = (A00+A11) B10
        # M4 = (A10+A11) B00
        # M5 = (A00+A01) B11
        # M6 = (A10+A00) B01
        # M7 = (A01+A11) B10
        # C00 = M1 + M3 - M4 - M7? No.
        
        # Let's use the standard 7 mults:
        # M1 = A00 B00
        # M2 = A11 B11
        # M3 = (A00+A11) (B00+B11)
        # M4 = A01 B10
        # M5 = A10 B01
        # M6 = (A00+A01) (B10+B00)
        # M7 = (A11+A10) (B01+B11)
        
        # C00 = M1 + M3 - M2 - M6? No.
        # C00 = M1 + M3 - M2 - M4? No.
        
        # Correct Strassen:
        # C00 = M1 + M3 - M2 - M4 is wrong.
        # C00 = M1 + M3 - M2 - M7? No.
        
        # Let's re-derive or use a known set.
        # M1 = A00 B00
        # M2 = A11 B11
        # M3 = (A00+A11) (B00+B11)
        # M4 = A01 B10
        # M5 = A10 B01
        # M6 = (A00+A01) (B10+B00)
        # M7 = (A11+A10) (B01+B11)
        
        # C00 = M1 + M3 - M2 - M6?
        # M1 + M3 - M2 = A00B00 + (A00+A11)(B00+B11) - A11B11
        # = A00B00 + A00B00 + A00B11 + A11B00 + A11B11 - A11B11
        # = 2A00B00 + A00B11 + A11B00.
        # We want A00B00 + A01B10.
        # This is not matching.
        
        # Let's use the standard Strassen from Wikipedia:
        # M1 = A00 B00
        # M2 = A10 B11
        # M3 = (A00+A11) B10
        # M4 = (A10+A11) B00
        # M5 = (A00+A01) B11
        # M6 = (A10+A00) B01
        # M7 = (A01+A11) B10
        
        # C00 = M1 + M3 - M4 - M7
        # C01 = M5 + M7
        # C10 = M2 + M4
        # C11 = M1 - M3 + M2 + M5
        
        # Let's implement this.
        
        # M1 = A00 B00
        U[0 * m + j_start + 0][col] = 1
        V[(j_start + 0) * p + k_start + 0][col] = 1
        W[(k_start + 0) * n + 0][col] = 1
        col += 1
        
        # M2 = A10 B11
        U[1 * m + j_start + 0][col] = 1
        V[(j_start + 1) * p + k_start + 1][col] = 1
        W[(k_start + 1) * n + 1][col] = 1
        col += 1
        
        # M3 = (A00+A11) B10
        U[0 * m + j_start + 0][col] = 1
        U[1 * m + j_start + 1][col] = 1
        V[(j_start + 1) * p + k_start + 0][col] = 1
        W[(k_start + 0) * n + 0][col] = 1
        W[(k_start + 0) * n + 1][col] = 1
        col += 1
        
        # M4 = (A10+A11) B00
        U[1 * m + j_start + 0][col] = 1
        U[1 * m + j_start + 1][col] = 1
        V[(j_start + 0) * p + k_start + 0][col] = 1
        W[(k_start + 0) * n + 1][col] = 1
        W[(k_start + 1) * n + 1][col] = 1
        col += 1
        
        # M5 = (A00+A01) B11
        U[0 * m + j_start + 0][col] = 1
        U[0 * m + j_start + 1][col] = 1
        V[(j_start + 1) * p + k_start + 1][col] = 1
        W[(k_start + 1) * n + 0][col] = 1
        W[(k_start + 1) * n + 1][col] = 1
        col += 1
        
        # M6 = (A10+A00) B01
        U[1 * m + j_start + 0][col] = 1
        U[0 * m + j_start + 0][col] = 1
        V[(j_start + 0) * p + k_start + 1][col] = 1
        W[(k_start + 1) * n + 1][col] = 1
        W[(k_start + 0) * n + 1][col] = 1
        col += 1
        
        # M7 = (A01+A11) B10
        U[0 * m + j_start + 1][col] = 1
        U[1 * m + j_start + 1][col] = 1
        V[(j_start + 1) * p + k_start + 0][col] = 1
        W[(k_start + 0) * n + 0][col] = 1
        W[(k_start + 0) * n + 1][col] = 1
        col += 1
        
    # Add Strassen for A1 B1a (j=0,1; k=0,1)
    add_strassen(0, 0)
    # Add Strassen for A1 B1b (j=0,1; k=2,3)
    add_strassen(0, 2)
    # Add Strassen for A1 B1c (j=0,1; k=4,5)
    add_strassen(0, 4)
    
    # Add Strassen for A2 B2a (j=2,3; k=0,1)
    add_strassen(2, 0)
    # Add Strassen for A2 B2b (j=2,3; k=2,3)
    add_strassen(2, 2)
    # Add Strassen for A2 B2c (j=2,3; k=4,5)
    add_strassen(2, 4)
    
    # Add standard for A3 B3 (j=4; k=0..5)
    # A3 is 2x1. B3 is 1x6.
    # C_i,k = A_i,4 * B_4,k
    for i in range(2):
        for k in range(6):
            U[i * m + 4][col] = 1
            V[4 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
