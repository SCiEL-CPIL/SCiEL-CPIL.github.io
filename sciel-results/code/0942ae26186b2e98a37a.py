import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 42
    rank = 42
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen decomposition for 2x2 * 2x2
    # A is 2x2, B is 2x2, C is 2x2
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def add_strassen(a_offset, b_offset, c_offset, sign_a=1, sign_b=1):
        nonlocal col
        # a_offset: (i_row, j_col) start for A block
        # b_offset: (j_row, k_col) start for B block
        # c_offset: (k_row, i_col) start for C block (W index is k*n + i)
        
        # Indices in A: i in 0..1, j in 0..1
        # Indices in B: j in 0..1, k in 0..1
        # Indices in C: i in 0..1, k in 0..1
        
        # We need to map the 7 multiplications to the global U, V, W.
        # U[i*m + j, col]
        # V[j*p + k, col]
        # W[k*n + i, col]
        
        # M1
        U[a_offset[0]*m + a_offset[1]][col] = sign_a
        U[(a_offset[0]+1)*m + (a_offset[1]+1)][col] = sign_a
        V[b_offset[0]*p + b_offset[1]][col] = sign_b
        V[(b_offset[0]+1)*p + (b_offset[1]+1)][col] = sign_b
        W[c_offset[0]*n + c_offset[1]][col] = 1 # C00
        W[(c_offset[0]+1)*n + c_offset[1]][col] = 1 # C11
        col += 1
        
        # M2
        U[(a_offset[0]+1)*m + a_offset[1]][col] = sign_a
        U[(a_offset[0]+1)*m + (a_offset[1]+1)][col] = sign_a
        V[b_offset[0]*p + b_offset[1]][col] = sign_b
        W[(c_offset[0]+1)*n + c_offset[1]][col] = 1 # C10
        col += 1
        
        # M3
        U[a_offset[0]*m + a_offset[1]][col] = sign_a
        V[b_offset[0]*p + (b_offset[1]+1)][col] = sign_b
        V[(b_offset[0]+1)*p + (b_offset[1]+1)][col] = -sign_b
        W[c_offset[0]*n + (c_offset[1]+1)][col] = 1 # C01
        W[(c_offset[0]+1)*n + (c_offset[1]+1)][col] = 1 # C11
        col += 1
        
        # M4
        U[(a_offset[0]+1)*m + (a_offset[1]+1)][col] = sign_a
        V[(b_offset[0]+1)*p + b_offset[1]][col] = sign_b
        V[b_offset[0]*p + b_offset[1]][col] = -sign_b
        W[c_offset[0]*n + c_offset[1]][col] = 1 # C00
        W[(c_offset[0]+1)*n + c_offset[1]][col] = 1 # C10
        col += 1
        
        # M5
        U[a_offset[0]*m + a_offset[1]][col] = sign_a
        U[a_offset[0]*m + (a_offset[1]+1)][col] = sign_a
        V[(b_offset[0]+1)*p + (b_offset[1]+1)][col] = sign_b
        W[c_offset[0]*n + c_offset[1]][col] = -1 # C00
        W[c_offset[0]*n + (c_offset[1]+1)][col] = 1 # C01
        col += 1
        
        # M6
        U[(a_offset[0]+1)*m + a_offset[1]][col] = sign_a
        U[a_offset[0]*m + a_offset[1]][col] = -sign_a
        V[b_offset[0]*p + b_offset[1]][col] = sign_b
        V[b_offset[0]*p + (b_offset[1]+1)][col] = sign_b
        W[(c_offset[0]+1)*n + (c_offset[1]+1)][col] = 1 # C11
        col += 1
        
        # M7
        U[a_offset[0]*m + (a_offset[1]+1)][col] = sign_a
        U[(a_offset[0]+1)*m + (a_offset[1]+1)][col] = -sign_a
        V[(b_offset[0]+1)*p + b_offset[1]][col] = sign_b
        V[(b_offset[0]+1)*p + (b_offset[1]+1)][col] = sign_b
        W[c_offset[0]*n + c_offset[1]][col] = 1 # C00
        col += 1
        
    # Part 1: C_1 (cols 0,1 of C)
    # A_1 (rows 0,1 of A, cols 0,1) * B_11 (rows 0,1 of B, cols 0,1) -> C_11 (rows 0,1 of C, cols 0,1)
    add_strassen((0,0), (0,0), (0,0))
    
    # A_2 (rows 0,1 of A, cols 2,3) * B_21 (rows 2,3 of B, cols 0,1) -> C_12 (rows 0,1 of C, cols 0,1)
    add_strassen((0,2), (2,0), (0,0))
    
    # Part 2: C_2 (cols 2,3,4,5 of C)
    # Split into C_2L (cols 2,3) and C_2R (cols 4,5)
    
    # C_2L = A * B_L2
    # A_1 * B_1L2 (rows 0,1 of B, cols 2,3) -> C_2L1 (rows 0,1 of C, cols 2,3)
    add_strassen((0,0), (0,2), (2,0))
    
    # A_2 * B_2L2 (rows 2,3 of B, cols 2,3) -> C_2L2 (rows 0,1 of C, cols 2,3)
    add_strassen((0,2), (2,2), (2,0))
    
    # C_2R = A * B_R2
    # A_1 * B_1R2 (rows 0,1 of B, cols 4,5) -> C_2R1 (rows 0,1 of C, cols 4,5)
    add_strassen((0,0), (0,4), (4,0))
    
    # A_2 * B_2R2 (rows 2,3 of B, cols 4,5) -> C_2R2 (rows 0,1 of C, cols 4,5)
    add_strassen((0,2), (2,4), (4,0))
    
    return {"U": U, "V": V, "W": W}
