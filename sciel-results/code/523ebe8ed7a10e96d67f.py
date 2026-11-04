import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 5
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 product
    # A is 2x2 at (r0, c0), B is 2x2 at (r0, c0) in B matrix
    # C is 2x2 at (r0, c0) in C matrix
    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # A indices: (a_r0, a_c0), (a_r0, a_c0+1), (a_r0+1, a_c0), (a_r0+1, a_c0+1)
        # B indices: (b_r0, b_c0), (b_r0, b_c0+1), (b_r0+1, b_c0), (b_r0+1, b_c0+1)
        
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We define the terms (u, v, w) where w is the coefficient for C element
        # u is linear form on A, v on B.
        
        # Term 1: M1
        # u: A11 + A22 -> U[a_r0, a_c0]=1, U[a_r0+1, a_c0+1]=1
        # v: B11 + B22 -> V[b_r0, b_c0]=1, V[b_r0+1, b_c0+1]=1
        # w: C11=1, C22=1
        U[a_r0*m + a_c0][col] = 1
        U[(a_r0+1)*m + (a_c0+1)][col] = 1
        V[b_r0*p + b_c0][col] = 1
        V[(b_r0+1)*p + (b_c0+1)][col] = 1
        W[c_c0*n + c_r0][col] = 1      # C11
        W[(c_c0+1)*n + (c_r0+1)][col] = 1 # C22
        col += 1
        
        # Term 2: M2
        # u: A21 + A22
        # v: B11
        # w: C21=1
        U[(a_r0+1)*m + a_c0][col] = 1
        U[(a_r0+1)*m + (a_c0+1)][col] = 1
        V[b_r0*p + b_c0][col] = 1
        W[c_c0*n + (c_r0+1)][col] = 1 # C21
        col += 1
        
        # Term 3: M3
        # u: A11
        # v: B12 - B22
        # w: C12=1
        U[a_r0*m + a_c0][col] = 1
        V[b_r0*p + (b_c0+1)][col] = 1
        V[(b_r0+1)*p + (b_c0+1)][col] = -1
        W[(c_c0+1)*n + c_r0][col] = 1 # C12
        col += 1
        
        # Term 4: M4
        # u: A22
        # v: B21 - B11
        # w: C11=1, C21=1
        U[(a_r0+1)*m + (a_c0+1)][col] = 1
        V[(b_r0+1)*p + b_c0][col] = 1
        V[b_r0*p + b_c0][col] = -1
        W[c_c0*n + c_r0][col] = 1     # C11
        W[c_c0*n + (c_r0+1)][col] = 1 # C21
        col += 1
        
        # Term 5: M5
        # u: A11 + A12
        # v: B22
        # w: C11=-1, C12=1
        U[a_r0*m + a_c0][col] = 1
        U[a_r0*m + (a_c0+1)][col] = 1
        V[(b_r0+1)*p + (b_c0+1)][col] = 1
        W[c_c0*n + c_r0][col] = -1    # C11
        W[(c_c0+1)*n + c_r0][col] = 1 # C12
        col += 1
        
        # Term 6: M6
        # u: A21 - A11
        # v: B11 + B12
        # w: C22=1
        U[(a_r0+1)*m + a_c0][col] = 1
        U[a_r0*m + a_c0][col] = -1
        V[b_r0*p + b_c0][col] = 1
        V[b_r0*p + (b_c0+1)][col] = 1
        W[(c_c0+1)*n + (c_r0+1)][col] = 1 # C22
        col += 1
        
        # Term 7: M7
        # u: A12 - A22
        # v: B21 + B22
        # w: C11=1
        U[a_r0*m + (a_c0+1)][col] = 1
        U[(a_r0+1)*m + (a_c0+1)][col] = -1
        V[(b_r0+1)*p + b_c0][col] = 1
        V[(b_r0+1)*p + (b_c0+1)][col] = 1
        W[c_c0*n + c_r0][col] = 1 # C11
        col += 1

    # Helper to add a 2x2x1 product
    # A is 2x2 at (r0, c0), b is 2x1 at (r0, c0) in B matrix
    # C is 2x1 at (r0, c0) in C matrix
    def add_2x2x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # C1 = A11 b1 + A12 b2
        # C2 = A21 b1 + A22 b2
        
        # Term 1: A11 b1 -> C1
        U[a_r0*m + a_c0][col] = 1
        V[b_r0*p + b_c0][col] = 1
        W[c_c0*n + c_r0][col] = 1
        col += 1
        
        # Term 2: A12 b2 -> C1
        U[a_r0*m + (a_c0+1)][col] = 1
        V[(b_r0+1)*p + b_c0][col] = 1
        W[c_c0*n + c_r0][col] = 1
        col += 1
        
        # Term 3: A21 b1 -> C2
        U[(a_r0+1)*m + a_c0][col] = 1
        V[b_r0*p + b_c0][col] = 1
        W[c_c0*n + (c_r0+1)][col] = 1
        col += 1
        
        # Term 4: A22 b2 -> C2
        U[(a_r0+1)*m + (a_c0+1)][col] = 1
        V[(b_r0+1)*p + b_c0][col] = 1
        W[c_c0*n + (c_r0+1)][col] = 1
        col += 1

    # Helper to add a 1x4x4 product
    # A is 1x4 at (r0, c0), B is 4x4 at (r0, c0)
    # C is 1x4 at (r0, c0)
    def add_1x4x4(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        for j in range(4):
            for k in range(4):
                U[a_r0*m + (a_c0+j)][col] = 1
                V[(b_r0+j)*p + (b_c0+k)][col] = 1
                W[(c_c0+k)*n + c_r0][col] = 1
                col += 1

    # Helper to add a 1x4x1 product
    # A is 1x4 at (r0, c0), B is 4x1 at (r0, c0)
    # C is 1x1 at (r0, c0)
    def add_1x4x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        for j in range(4):
            U[a_r0*m + (a_c0+j)][col] = 1
            V[(b_r0+j)*p + b_c0][col] = 1
            W[c_c0*n + c_r0][col] = 1
            col += 1

    # Block 1: A_top (2x4) * B_left (4x4) -> C_top_left (2x4)
    # A_top: rows 0,1. Cols 0-3.
    # B_left: rows 0-3. Cols 0-3.
    # C_top_left: rows 0,1. Cols 0-3.
    
    # Split A_top into A1 (cols 0,1) and A2 (cols 2,3)
    # Split B_left into B11 (rows 0,1 cols 0,1), B12 (rows 0,1 cols 2,3), B21 (rows 2,3 cols 0,1), B22 (rows 2,3 cols 2,3)
    
    # P1: A1 * B11 -> C cols 0,1
    add_strassen(0, 0, 0, 0, 0, 0)
    # P2: A2 * B21 -> C cols 0,1
    add_strassen(0, 2, 2, 0, 0, 0)
    # P3: A1 * B12 -> C cols 2,3
    add_strassen(0, 0, 0, 2, 0, 2)
    # P4: A2 * B22 -> C cols 2,3
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # Block 2: A_top (2x4) * B_right (4x1) -> C_top_right (2x1)
    # B_right: col 4.
    # Split B_right into b1 (rows 0,1) and b2 (rows 2,3)
    
    # P5: A1 * b1 -> C col 4
    add_2x2x1(0, 0, 0, 4, 0, 4)
    # P6: A2 * b2 -> C col 4
    add_2x2x1(0, 2, 2, 4, 0, 4)
    
    # Block 3: A_bot (1x4) * B_left (4x4) -> C_bot_left (1x4)
    # A_bot: row 2.
    # P7: 1x4x4
    add_1x4x4(2, 0, 0, 0, 2, 0)
    
    # Block 4: A_bot (1x4) * B_right (4x1) -> C_bot_right (1x1)
    # P8: 1x4x1
    add_1x4x1(2, 0, 0, 4, 2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
