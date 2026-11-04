"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    
    # Rank 46 decomposition
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Function to add a 2x2x2 Strassen decomposition
    # A block at rows [a0, a0+1], cols [a1, a1+1] of A (global)
    # B block at rows [b0, b0+1], cols [b1, b1+1] of B (global)
    # C block at rows [c0, c0+1], cols [c1, c1+1] of C (global)
    # Note: C indices are k*n + i.
    # A indices are i*m + j.
    # B indices are j*p + k.
    
    def add_strassen(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        # a_row, a_col: start of A block (2x2)
        # b_row, b_col: start of B block (2x2)
        # c_row, c_col: start of C block (2x2)
        
        # M1 = (A11 + A22) * (B11 + B22)
        U[a_row*m + a_col][col] = 1
        U[(a_row+1)*m + (a_col+1)][col] = 1
        V[b_row*p + b_col][col] = 1
        V[(b_row+1)*p + (b_col+1)][col] = 1
        W[c_row*n + c_col][col] = 1
        W[(c_row+1)*n + (c_col+1)][col] = 1
        col += 1
        
        # M2 = (A21 + A22) * B11
        U[(a_row+1)*m + a_col][col] = 1
        U[(a_row+1)*m + (a_col+1)][col] = 1
        V[b_row*p + b_col][col] = 1
        W[(c_row+1)*n + c_col][col] = 1
        W[(c_row+1)*n + (c_col+1)][col] = 1
        col += 1
        
        # M3 = A11 * (B12 - B22)
        U[a_row*m + a_col][col] = 1
        V[b_row*p + (b_col+1)][col] = 1
        V[(b_row+1)*p + (b_col+1)][col] = -1
        W[c_row*n + (c_col+1)][col] = 1
        W[(c_row+1)*n + (c_col+1)][col] = 1
        col += 1
        
        # M4 = A22 * (B21 - B11)
        U[(a_row+1)*m + (a_col+1)][col] = 1
        V[(b_row+1)*p + b_col][col] = 1
        V[b_row*p + b_col][col] = -1
        W[(c_row+1)*n + c_col][col] = 1
        W[(c_row+1)*n + (c_col+1)][col] = 1
        col += 1
        
        # M5 = (A11 + A12) * B22
        U[a_row*m + a_col][col] = 1
        U[a_row*m + (a_col+1)][col] = 1
        V[(b_row+1)*p + (b_col+1)][col] = 1
        W[(c_row+1)*n + c_col][col] = -1
        W[(c_row+1)*n + (c_col+1)][col] = 1
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[(a_row+1)*m + a_col][col] = 1
        U[a_row*m + a_col][col] = -1
        V[b_row*p + b_col][col] = 1
        V[b_row*p + (b_col+1)][col] = 1
        W[(c_row+1)*n + (c_col+1)][col] = 1
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[a_row*m + (a_col+1)][col] = 1
        U[(a_row+1)*m + (a_col+1)][col] = -1
        V[(b_row+1)*p + b_col][col] = 1
        V[(b_row+1)*p + (b_col+1)][col] = 1
        W[c_row*n + c_col][col] = 1
        W[(c_row+1)*n + (c_col+1)][col] = 1
        col += 1

    # Function to add a 2x2x1 schoolbook decomposition
    # A block (2x2), B block (2x1), C block (2x1)
    def add_schoolbook_2x2x1(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        # C1 = A11*B1 + A12*B2
        # C2 = A21*B1 + A22*B2
        
        # Term 1: A11 * B1 -> C1
        U[a_row*m + a_col][col] = 1
        V[b_row*p + b_col][col] = 1
        W[c_row*n + c_col][col] = 1
        col += 1
        
        # Term 2: A12 * B2 -> C1
        U[a_row*m + (a_col+1)][col] = 1
        V[(b_row+1)*p + b_col][col] = 1
        W[c_row*n + c_col][col] = 1
        col += 1
        
        # Term 3: A21 * B1 -> C2
        U[(a_row+1)*m + a_col][col] = 1
        V[b_row*p + b_col][col] = 1
        W[(c_row+1)*n + c_col][col] = 1
        col += 1
        
        # Term 4: A22 * B2 -> C2
        U[(a_row+1)*m + (a_col+1)][col] = 1
        V[(b_row+1)*p + b_col][col] = 1
        W[(c_row+1)*n + c_col][col] = 1
        col += 1

    # Block 1: A1 (rows 0-1, cols 0-1), B1 (rows 0-1, cols 0-6)
    # Split B1 into B11 (cols 0-1), B12 (cols 2-3), B13 (cols 4-5), B14 (cols 6)
    # C1 (rows 0-1, cols 0-6)
    
    # B11: 2x2
    add_strassen(0, 0, 0, 0, 0, 0)
    # B12: 2x2
    add_strassen(0, 0, 0, 2, 0, 2)
    # B13: 2x2
    add_strassen(0, 0, 0, 4, 0, 4)
    # B14: 2x1
    add_schoolbook_2x2x1(0, 0, 0, 6, 0, 6)
    
    # Block 2: A2 (rows 0-1, cols 2-3), B2 (rows 2-3, cols 0-6)
    # Split B2 into B21 (cols 0-1), B22 (cols 2-3), B23 (cols 4-5), B24 (cols 6)
    # C2 (rows 0-1, cols 0-6)
    
    # B21: 2x2
    add_strassen(0, 2, 2, 0, 0, 0)
    # B22: 2x2
    add_strassen(0, 2, 2, 2, 0, 2)
    # B23: 2x2
    add_strassen(0, 2, 2, 4, 0, 4)
    # B24: 2x1
    add_schoolbook_2x2x1(0, 2, 2, 6, 0, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
