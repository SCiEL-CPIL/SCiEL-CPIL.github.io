"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 39 construction: 5 pairs of columns (Strassen rank 7) + 1 single column (Schoolbook rank 4)
    rank = 5 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Process 5 pairs of columns
    for t in range(5):
        c0 = 2 * t
        c1 = 2 * t + 1
        
        # Strassen Products for block t
        # P1: (A00+A11)(B0c0+B1c1)
        U[0][col_offset] = 1
        U[3][col_offset] = 1
        V[c0][col_offset] = 1
        V[11 + c1][col_offset] = 1
        W[2 * c0][col_offset] = 1
        W[2 * c1 + 1][col_offset] = 1
        col_offset += 1
        
        # P2: (A10+A11)B0c0
        U[2][col_offset] = 1
        U[3][col_offset] = 1
        V[c0][col_offset] = 1
        W[2 * c0 + 1][col_offset] = 1
        W[2 * c1 + 1][col_offset] = -1
        col_offset += 1
        
        # P3: A00(B0c1-B1c1)
        U[0][col_offset] = 1
        V[c1][col_offset] = 1
        V[11 + c1][col_offset] = -1
        W[2 * c1][col_offset] = 1
        W[2 * c1 + 1][col_offset] = 1
        col_offset += 1
        
        # P4: A11(B1c0-B0c0)
        U[3][col_offset] = 1
        V[11 + c0][col_offset] = 1
        V[c0][col_offset] = -1
        W[2 * c0][col_offset] = 1
        W[2 * c0 + 1][col_offset] = 1
        col_offset += 1
        
        # P5: (A00+A01)B1c1
        U[0][col_offset] = 1
        U[1][col_offset] = 1
        V[11 + c1][col_offset] = 1
        W[2 * c0][col_offset] = -1
        W[2 * c1][col_offset] = 1
        col_offset += 1
        
        # P6: (A10-A00)(B0c0+B0c1)
        U[2][col_offset] = 1
        U[0][col_offset] = -1
        V[c0][col_offset] = 1
        V[c1][col_offset] = 1
        W[2 * c1 + 1][col_offset] = 1
        col_offset += 1
        
        # P7: (A01-A11)(B1c0+B1c1)
        U[1][col_offset] = 1
        U[3][col_offset] = -1
        V[11 + c0][col_offset] = 1
        V[11 + c1][col_offset] = 1
        W[2 * c0][col_offset] = 1
        col_offset += 1

    # Process single column 10
    c0 = 10
    
    # Q1: A00 B0c0
    U[0][col_offset] = 1
    V[c0][col_offset] = 1
    W[2 * c0][col_offset] = 1
    col_offset += 1
    
    # Q2: A01 B1c0
    U[1][col_offset] = 1
    V[11 + c0][col_offset] = 1
    W[2 * c0][col_offset] = 1
    col_offset += 1
    
    # Q3: A10 B0c0
    U[2][col_offset] = 1
    V[c0][col_offset] = 1
    W[2 * c0 + 1][col_offset] = 1
    col_offset += 1
    
    # Q4: A11 B1c0
    U[3][col_offset] = 1
    V[11 + c0][col_offset] = 1
    W[2 * c0 + 1][col_offset] = 1
    col_offset += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
