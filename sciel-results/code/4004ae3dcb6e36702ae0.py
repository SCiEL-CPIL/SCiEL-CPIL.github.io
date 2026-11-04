"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    # Rank 44 decomposition using Strassen for 2x2 blocks
    # A is 3x4, B is 4x4, C is 3x4
    # Split A into 2x2 (top), 1x2 (bottom)
    # Split B into 2x2 (top-left), 2x2 (top-right), 2x2 (bottom-left), 2x2 (bottom-right)
    # Split C into 2x2 (top), 1x2 (bottom)
    
    # Indices for A: 0..2 rows, 0..3 cols
    # Indices for B: 0..3 rows, 0..3 cols
    # Indices for C: 0..2 rows, 0..3 cols
    
    # Blocks:
    # A11: rows 0,1; cols 0,1
    # A12: rows 0,1; cols 2,3
    # A21: row 2; cols 0,1
    # A22: row 2; cols 2,3
    
    # B11: rows 0,1; cols 0,1
    # B12: rows 0,1; cols 2,3
    # B21: rows 2,3; cols 0,1
    # B22: rows 2,3; cols 2,3
    
    # C11: rows 0,1; cols 0,1
    # C12: rows 0,1; cols 2,3
    # C21: row 2; cols 0,1
    # C22: row 2; cols 2,3
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen terms for 2x2 * 2x2
    def add_strassen(A_block, B_block, C_block, offset):
        # A_block: list of (row, col) for 2x2
        # B_block: list of (row, col) for 2x2
        # C_block: list of (row, col) for 2x2
        # offset: current column index
        nonlocal col
        # M1
        U[A_block[0][0]*m + A_block[0][1]][col] = 1
        U[A_block[1][0]*m + A_block[1][1]][col] = 1
        V[B_block[0][0]*p + B_block[0][1]][col] = 1
        V[B_block[1][0]*p + B_block[1][1]][col] = 1
        W[C_block[0][0]*n + C_block[0][1]][col] = 1
        W[C_block[1][0]*n + C_block[1][1]][col] = 1
        col += 1
        
        # M2
        U[A_block[1][0]*m + A_block[1][0]][col] = 1
        U[A_block[1][0]*m + A_block[1][1]][col] = 1
        V[B_block[0][0]*p + B_block[0][1]][col] = 1
        W[C_block[1][0]*n + C_block[1][1]][col] = 1
        W[C_block[1][0]*n + C_block[0][1]][col] = -1
        col += 1
        
        # M3
        U[A_block[0][0]*m + A_block[0][1]][col] = 1
        V[B_block[0][0]*p + B_block[1][1]][col] = 1
        V[B_block[1][0]*p + B_block[1][1]][col] = -1
        W[C_block[0][0]*n + C_block[1][1]][col] = 1
        W[C_block[1][0]*n + C_block[1][1]][col] = 1
        col += 1
        
        # M4
        U[A_block[1][0]*m + A_block[1][1]][col] = 1
        V[B_block[1][0]*p + B_block[1][1]][col] = 1
        V[B_block[0][0]*p + B_block[0][1]][col] = -1
        W[C_block[0][0]*n + C_block[0][1]][col] = 1
        W[C_block[1][0]*n + C_block[1][1]][col] = 1
        col += 1
        
        # M5
        U[A_block[0][0]*m + A_block[0][1]][col] = 1
        U[A_block[0][0]*m + A_block[1][1]][col] = 1
        V[B_block[1][0]*p + B_block[1][1]][col] = 1
        W[C_block[0][0]*n + C_block[0][1]][col] = -1
        W[C_block[0][0]*n + C_block[1][1]][col] = 1
        col += 1
        
        # M6
        U[A_block[1][0]*m + A_block[1][0]][col] = 1
        U[A_block[0][0]*m + A_block[0][1]][col] = -1
        V[B_block[0][0]*p + B_block[0][1]][col] = 1
        V[B_block[0][0]*p + B_block[1][1]][col] = 1
        W[C_block[1][0]*n + C_block[1][1]][col] = -1
        col += 1
        
        # M7
        U[A_block[0][0]*m + A_block[1][1]][col] = 1
        U[A_block[1][0]*m + A_block[1][1]][col] = -1
        V[B_block[1][0]*p + B_block[1][1]][col] = 1
        V[B_block[1][0]*p + B_block[0][1]][col] = 1
        W[C_block[0][0]*n + C_block[0][1]][col] = 1
        col += 1
        
    # Helper to add 1x2 * 2x2 terms
    def add_1x2(A_block, B_block, C_block, offset):
        # A_block: list of (row, col) for 1x2
        # B_block: list of (row, col) for 2x2
        # C_block: list of (row, col) for 1x2
        nonlocal col
        # T1: A0 * B00
        U[A_block[0][0]*m + A_block[0][1]][col] = 1
        V[B_block[0][0]*p + B_block[0][1]][col] = 1
        W[C_block[0][0]*n + C_block[0][1]][col] = 1
        col += 1
        
        # T2: A1 * B10
        U[A_block[0][0]*m + A_block[1][1]][col] = 1
        V[B_block[1][0]*p + B_block[1][1]][col] = 1
        W[C_block[0][0]*n + C_block[0][1]][col] = 1
        col += 1
        
        # T3: A0 * B01
        U[A_block[0][0]*m + A_block[0][1]][col] = 1
        V[B_block[0][0]*p + B_block[1][1]][col] = 1
        W[C_block[0][0]*n + C_block[1][1]][col] = 1
        col += 1
        
        # T4: A1 * B11
        U[A_block[0][0]*m + A_block[1][1]][col] = 1
        V[B_block[1][0]*p + B_block[1][1]][col] = 1
        W[C_block[0][0]*n + C_block[1][1]][col] = 1
        col += 1

    # Define blocks
    A11 = [(0,0), (0,1), (1,0), (1,1)]
    A12 = [(0,2), (0,3), (1,2), (1,3)]
    A21 = [(2,0), (2,1)]
    A22 = [(2,2), (2,3)]
    
    B11 = [(0,0), (0,1), (1,0), (1,1)]
    B12 = [(0,2), (0,3), (1,2), (1,3)]
    B21 = [(2,0), (2,1), (3,0), (3,1)]
    B22 = [(2,2), (2,3), (3,2), (3,3)]
    
    C11 = [(0,0), (0,1), (1,0), (1,1)]
    C12 = [(0,2), (0,3), (1,2), (1,3)]
    C21 = [(2,0), (2,1)]
    C22 = [(2,2), (2,3)]
    
    # C11 = A11*B11 + A12*B21
    add_strassen(A11, B11, C11, col)
    add_strassen(A12, B21, C11, col)
    
    # C12 = A11*B12 + A12*B22
    add_strassen(A11, B12, C12, col)
    add_strassen(A12, B22, C12, col)
    
    # C21 = A21*B11 + A22*B21
    add_1x2(A21, B11, C21, col)
    add_1x2(A22, B21, C21, col)
    
    # C22 = A21*B12 + A22*B22
    add_1x2(A21, B12, C22, col)
    add_1x2(A22, B22, C22, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
