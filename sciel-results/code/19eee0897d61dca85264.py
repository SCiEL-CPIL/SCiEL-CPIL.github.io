"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x2 * 2x13
    # Group B into 6 pairs of columns (each pair is 2x2) plus 1 single column.
    # For each 2x2 block: Strassen (7 terms). For the last column: schoolbook (4 terms).
    # Total rank = 6*7 + 4 = 46.
    
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]   # shape (4, 46)
    V = [[0] * rank for _ in range(m * p)]   # shape (26, 46)
    W = [[0] * rank for _ in range(p * n)]   # shape (26, 46)
    
    col = 0
    # Process 6 pairs of columns: columns (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for block in range(6):
        c0 = 2 * block
        c1 = 2 * block + 1
        
        # B indices: row 0, col c0 -> 0*p + c0 = c0; row 0, col c1 -> c1
        #              row 1, col c0 -> 1*p + c0 = 13 + c0; row 1, col c1 -> 13 + c1
        # C indices (k*n + i): C[0,c0] = c0*2+0 = 2*c0; C[1,c0] = c0*2+1 = 2*c0+1
        #                       C[0,c1] = c1*2+0 = 2*c1; C[1,c1] = c1*2+1 = 2*c1+1
        
        # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        # B indices within block: B[0,c0]=c0, B[0,c1]=c1, B[1,c0]=13+c0, B[1,c1]=13+c1
        # C indices: C[0,c0]=2*c0, C[1,c0]=2*c0+1, C[0,c1]=2*c1, C[1,c1]=2*c1+1
        
        # Strassen 7 terms for 2x2 * 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00+B01)
        # M3 = (A00-A01)(B10+B11)
        # M4 = (A11-A10)(B01+B11)
        # M5 = (A00+A11)(B00+B01)
        # M6 = (A00-A01)(B00+B10)
        # M7 = (A10+A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        b00 = c0
        b01 = c1
        b10 = 13 + c0
        b11 = 13 + c1
        
        c00 = 2 * c0
        c10 = 2 * c0 + 1
        c01 = 2 * c1
        c11 = 2 * c1 + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[0][col] = 1; U[3][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        
        # M2 = (A10+A11)(B00+B01)
        U[2][col] = 1; U[3][col] = 1
        V[b00][col] = 1; V[b01][col] = 1
        W[c10][col] = 1
        col += 1
        
        # M3 = (A00-A01)(B10+B11)
        U[0][col] = 1; U[1][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c01][col] = 1
        col += 1
        
        # M4 = (A11-A10)(B01+B11)
        U[3][col] = 1; U[2][col] = -1
        V[b01][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c10][col] = 1
        col += 1
        
        # M5 = (A00+A11)(B00+B01)
        U[0][col] = 1; U[3][col] = 1
        V[b00][col] = 1; V[b01][col] = 1
        W[c01][col] = 1; W[c00][col] = -1
        col += 1
        
        # M6 = (A00-A01)(B00+B10)
        U[0][col] = 1; U[1][col] = -1
        V[b00][col] = 1; V[b10][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M7 = (A10+A11)(B10+B11)
        U[2][col] = 1; U[3][col] = 1
        V[b10][col] = 1; V[b11][col] = 1
        W[c00][col] = 1
        col += 1
    
    # Last single column (column 12): schoolbook 4 terms
    c = 12
    b0 = 12      # B[0,12]
    b1 = 25      # B[1,12] = 13+12
    c00 = 24     # C[0,12] = 12*2+0
    c10 = 25     # C[1,12] = 12*2+1
    
    # A00 * B00 -> C00
    U[0][col] = 1
    V[b0][col] = 1
    W[c00][col] = 1
    col += 1
    
    # A01 * B00 -> C00... wait, A is 2x2, B is 2x1.
    # C[0,12] = A[0,0]*B[0,12] + A[0,1]*B[1,12]
    # C[1,12] = A[1,0]*B[0,12] + A[1,1]*B[1,12]
    
    # A01 * B11 -> C00 (A[0,1] is index 1, B[1,12] is index 25)
    U[1][col] = 1
    V[b1][col] = 1
    W[c00][col] = 1
    col += 1
    
    # A10 * B00 -> C10
    U[2][col] = 1
    V[b0][col] = 1
    W[c10][col] = 1
    col += 1
    
    # A11 * B11 -> C10
    U[3][col] = 1
    V[b1][col] = 1
    W[c10][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
