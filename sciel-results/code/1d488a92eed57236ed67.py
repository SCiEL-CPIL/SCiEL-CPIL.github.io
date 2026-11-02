"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=4
    rank = 14
    
    # U: (n*m, rank) = (4, 14)
    # V: (m*p, rank) = (8, 14)
    # W: (p*n, rank) = (8, 14)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Two independent Strassen 2x2x2 blocks
    # Block 0: columns 0,1 of the 4-column matrix
    # Block 1: columns 2,3 of the 4-column matrix
    
    for block in range(2):
        base = block * 7
        # For block b, the b-matrix columns are (2*b, 2*b+1)
        # Strassen bilinear forms:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b01-b11)
        # m4 = a11(b10-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b10)
        # m7 = (a11-a01)(b01+b11)  -- wait, let me use standard Strassen
        
        # Standard Strassen:
        # m1 = (A00+A11)(B00+B11)
        # m2 = (A10+A11)(B00)
        # m3 = (A00)(B01-B11)
        # m4 = (A11)(B10-B00)
        # m5 = (A00+A01)(B11)
        # m6 = (A10-A00)(B00+B10)
        # m7 = (A01-A11)(B01+B11)
        
        # Result:
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        col_offset = 2 * block  # column offset in the 4-column matrix
        
        # m1: (A00+A11)(B00+B11)
        U[0][base + 0] = 1  # A00
        U[3][base + 0] = 1  # A11
        V[0][base + 0] = 1  # B00 (row 0, col 0)
        V[3][base + 0] = 1  # B11 (row 1, col 1) -> m*p index: 1*2+1=3
        # C00 += m1: W[0*2+0] = W[0]
        W[0][base + 0] = 1
        # C11 += m1: W[1*2+1] = W[3]
        W[3][base + 0] = 1
        
        # m2: (A10+A11)(B00)
        U[2][base + 1] = 1  # A10
        U[3][base + 1] = 1  # A11
        V[0][base + 1] = 1  # B00
        # C10 += m2: W[0*2+1] = W[1]
        W[1][base + 1] = 1
        # C11 -= m2: W[1*2+1] = W[3]
        W[3][base + 1] = -1
        
        # m3: A00(B01-B11)
        U[0][base + 2] = 1  # A00
        V[1][base + 2] = 1  # B01 (row 0, col 1) -> index 0*2+1=1
        V[3][base + 2] = -1 # B11 (row 1, col 1) -> index 1*2+1=3
        # C01 += m3: W[1*2+0] = W[2]
        W[2][base + 2] = 1
        # C11 += m3: W[1*2+1] = W[3]
        W[3][base + 2] = 1
        
        # m4: A11(B10-B00)
        U[3][base + 3] = 1  # A11
        V[2][base + 3] = 1  # B10 (row 1, col 0) -> index 1*2+0=2
        V[0][base + 3] = -1 # B00
        # C00 += m4: W[0*2+0] = W[0]
        W[0][base + 3] = 1
        # C10 += m4: W[0*2+1] = W[1]
        W[1][base + 3] = 1
        
        # m5: (A00+A01)(B11)
        U[0][base + 4] = 1  # A00
        U[1][base + 4] = 1  # A01
        V[3][base + 4] = 1  # B11
        # C00 -= m5: W[0*2+0] = W[0]
        W[0][base + 4] = -1
        # C01 += m5: W[1*2+0] = W[2]
        W[2][base + 4] = 1
        
        # m6: (A10-A00)(B00+B10)
        U[2][base + 5] = 1  # A10
        U[0][base + 5] = -1 # A00
        V[0][base + 5] = 1  # B00
        V[2][base + 5] = 1  # B10
        # C11 += m6: W[1*2+1] = W[3]
        W[3][base + 5] = 1
        
        # m7: (A01-A11)(B01+B11)
        U[1][base + 6] = 1  # A01
        U[3][base + 6] = -1 # A11
        V[1][base + 6] = 1  # B01
        V[3][base + 6] = 1  # B11
        # C00 += m7: W[0*2+0] = W[0]
        W[0][base + 6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
