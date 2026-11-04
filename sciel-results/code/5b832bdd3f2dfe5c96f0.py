"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # C1 (2x2) from A1 (2x4) B1 (4x2) -> 14 mults
    # C2 (2x2) from A1 (2x4) B2 (4x2) -> 14 mults
    # C3, C4 (1x4) from A2 (1x4) B (4x4) -> 16 mults
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen decomposition
    # A rows a_start, a_start+1
    # B cols b_start, b_start+1
    # B rows b_row_start, b_row_start+1
    # C rows c_start, c_start+1
    # C cols c_col_start, c_col_start+1
    def add_strassen(a_r1, a_r2, b_c1, b_c2, b_r1, b_r2, c_r1, c_r2, c_c1, c_c2):
        nonlocal col
        # P1 = (A11+A22)(B11+B22)
        U[a_r1*m + b_c1][col] = 1
        U[a_r1*m + b_c2][col] = 1
        U[a_r2*m + b_c1][col] = 1
        U[a_r2*m + b_c2][col] = 1
        V[b_r1*p + b_c1][col] = 1
        V[b_r1*p + b_c2][col] = 1
        V[b_r2*p + b_c1][col] = 1
        V[b_r2*p + b_c2][col] = 1
        W[c_c1*n + c_r1][col] = 1
        W[c_c1*n + c_r2][col] = 1
        W[c_c2*n + c_r1][col] = 1
        W[c_c2*n + c_r2][col] = 1
        col += 1
        
        # P2 = (A21+A22)B11
        U[a_r2*m + b_c1][col] = 1
        U[a_r2*m + b_c2][col] = 1
        V[b_r1*p + b_c1][col] = 1
        W[c_c1*n + c_r2][col] = 1
        W[c_c2*n + c_r2][col] = 1
        col += 1
        
        # P3 = A11(B12-B22)
        U[a_r1*m + b_c1][col] = 1
        V[b_r1*p + b_c2][col] = 1
        V[b_r2*p + b_c2][col] = -1
        W[c_c2*n + c_r1][col] = 1
        col += 1
        
        # P4 = A22(B21-B11)
        U[a_r2*m + b_c2][col] = 1
        V[b_r2*p + b_c1][col] = 1
        V[b_r1*p + b_c1][col] = -1
        W[c_c1*n + c_r2][col] = 1
        col += 1
        
        # P5 = (A11+A12)B22
        U[a_r1*m + b_c1][col] = 1
        U[a_r1*m + b_c2][col] = 1
        V[b_r2*p + b_c2][col] = 1
        W[c_c2*n + c_r1][col] = 1
        col += 1
        
        # P6 = (A21-A11)(B11+B12)
        U[a_r2*m + b_c1][col] = 1
        U[a_r1*m + b_c1][col] = -1
        V[b_r1*p + b_c1][col] = 1
        V[b_r1*p + b_c2][col] = 1
        W[c_c1*n + c_r2][col] = 1
        W[c_c2*n + c_r2][col] = 1
        col += 1
        
        # P7 = (A12-A22)(B21+B22)
        U[a_r1*m + b_c2][col] = 1
        U[a_r2*m + b_c2][col] = -1
        V[b_r2*p + b_c1][col] = 1
        V[b_r2*p + b_c2][col] = 1
        W[c_c1*n + c_r1][col] = 1
        W[c_c2*n + c_r1][col] = 1
        col += 1

    # C1: A1 (rows 0,1) B1 (cols 0,1, rows 0,1) -> C1 (rows 0,1, cols 0,1)
    # Split B1 rows 0,1 and 2,3
    # A1 cols 0,1 and 2,3
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 2, 3, 2, 3, 0, 1, 0, 1)
    
    # C2: A1 (rows 0,1) B2 (cols 2,3, rows 0,1) -> C2 (rows 0,1, cols 2,3)
    # Split B2 rows 0,1 and 2,3
    # A1 cols 0,1 and 2,3
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1, 2, 3)
    add_strassen(0, 1, 2, 3, 2, 3, 0, 1, 2, 3)
    
    # C3, C4: A2 (row 2) B (cols 0,1,2,3, rows 0,1,2,3)
    # Rank 16
    for j in range(4):
        for k in range(4):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
