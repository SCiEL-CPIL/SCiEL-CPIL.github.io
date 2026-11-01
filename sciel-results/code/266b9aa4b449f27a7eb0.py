import time

DIMENSIONS = [2, 2, 12]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Use Strassen's algorithm for 2x2 matrix multiplication, applied to 6 column pairs
    # Total rank = 6 * 7 = 42
    rank = 42
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Correct Strassen matrices for 2x2 x 2x2
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    #
    # Strassen products:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    #
    # Output:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # U matrix: coefficients for A components in each Strassen product
    # Columns are P1..P7
    strassen_U = [
        [1,  0,  1,  0,  1,  -1,  0],  # A00
        [0,  0,  0,  0,  1,   0,  1],  # A01
        [0,  1,  0,  0,  0,   1,  0],  # A10
        [1,  1,  0,  1,  0,   0, -1],  # A11
    ]
    
    # V matrix: coefficients for B components in each Strassen product
    strassen_V = [
        [1,  1,  0, -1,  0,  1,  0],  # B00
        [0,  0,  1,  0,  0,  1,  0],  # B01
        [0,  0,  0,  1,  0,  0,  1],  # B10
        [1,  0, -1,  0,  1,  0,  1],  # B11
    ]
    
    # W matrix: coefficients for C components (how Strassen products combine)
    strassen_W = [
        [1,  0,  0,  1, -1,  0,  1],  # C00
        [0,  0,  1,  0,  1,  0,  0],  # C01
        [0,  1,  0,  1,  0,  0,  0],  # C10
        [1, -1, 1,  0,  0,  1,  0],  # C11
    ]
    
    # Fill in the matrices for each of the 6 column groups
    col = 0
    for group in range(6):
        for s in range(7):
            # U: A indices 0-3 (shared across all groups)
            for ai in range(4):
                U[ai][col] = strassen_U[ai][s]
            
            # V: B indices for this group
            # B index 0 = (j=0, l=2*group), B index 1 = (j=0, l=2*group+1)
            # B index 2 = (j=1, l=2*group), B index 3 = (j=1, l=2*group+1)
            for bi in range(4):
                j = bi // 2
                l_offset = bi % 2
                l = 2 * group + l_offset
                V[j * p + l][col] = strassen_V[bi][s]
            
            # W: C indices for this group
            # C index 0 = (i=0, l=2*group), C index 1 = (i=0, l=2*group+1)
            # C index 2 = (i=1, l=2*group), C index 3 = (i=1, l=2*group+1)
            for ci in range(4):
                i = ci // 2
                l_offset = ci % 2
                l = 2 * group + l_offset
                W[l * n + i][col] = strassen_W[ci][s]
            
            col += 1
        
        # Check deadline periodically
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
