"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 Strassen decomposition for 2x2 matrix multiplication.
    U: shape (4, 7)
    V: shape (4, 7)
    W: shape (4, 7)
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # C00 = P1 - P3 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P6
    
    # Indices:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # Term 1: P1
    U[0][0] = 1; U[3][0] = 1      # A00, A11
    V[0][0] = 1; V[3][0] = 1      # B00, B11
    W[0][0] = 1; W[3][0] = 1      # C00, C11
    
    # Term 2: P2
    U[2][1] = 1; U[3][1] = 1      # A10, A11
    V[0][1] = 1                  # B00
    W[1][1] = 1; W[3][1] = -1     # C10, C11
    
    # Term 3: P3
    U[0][2] = 1                  # A00
    V[1][2] = 1; V[3][2] = -1    # B01, B11
    W[0][2] = -1; W[2][2] = 1    # C00, C01
    
    # Term 4: P4
    U[3][3] = 1                  # A11
    V[2][3] = 1; V[0][3] = -1    # B10, B00
    W[0][3] = 1; W[1][3] = 1     # C00, C10
    
    # Term 5: P5
    U[0][4] = 1; U[1][4] = 1     # A00, A01
    V[3][4] = 1                  # B11
    W[0][4] = -1; W[2][4] = 1    # C00, C01
    
    # Term 6: P6
    U[2][5] = 1; U[0][5] = -1    # A10, A00
    V[0][5] = 1; V[1][5] = 1     # B00, B01
    W[3][5] = 1                  # C11
    
    # Term 7: P7
    U[1][6] = 1; U[3][6] = -1    # A01, A11
    V[2][6] = 1; V[3][6] = 1     # B10, B11
    W[0][6] = 1                  # C00

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
