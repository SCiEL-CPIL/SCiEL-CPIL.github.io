import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen(a_i, a_j, b_j, b_k, c_k, c_i):
        nonlocal col
        # A indices: (a_i, a_j)
        # B indices: (b_j, b_k)
        # C indices: (c_k, c_i)
        
        # M1 = (A11 + A22)(B11 + B22)
        U[a_i*4 + a_j][col] += 1
        U[a_i*4 + a_j+1][col] += 1
        V[b_j*7 + b_k][col] += 1
        V[b_j*7 + b_k+1][col] += 1
        W[c_k*2 + c_i][col] += 1
        W[c_k*2 + c_i+1][col] += 1
        col += 1
        
        # M2 = (A21 + A22)B11
        U[a_i*4 + a_j+1][col] += 1
        U[a_i*4 + a_j+2][col] += 1
        V[b_j*7 + b_k][col] += 1
        W[c_k*2 + c_i+1][col] += 1
        col += 1
        
        # M3 = A11(B12 - B22)
        U[a_i*4 + a_j][col] += 1
        V[b_j*7 + b_k+1][col] += 1
        V[b_j*7 + b_k+2][col] += -1
        W[c_k*2 + c_i+1][col] += 1
        col += 1
        
        # M4 = A22(B21 - B11)
        U[a_i*4 + a_j+1][col] += 1
        V[b_j*7 + b_k+1][col] += 1
        V[b_j*7 + b_k][col] += -1
        W[c_k*2 + c_i][col] += 1
        col += 1
        
        # M5 = (A11 + A12)B22
        U[a_i*4 + a_j][col] += 1
        U[a_i*4 + a_j+1][col] += 1
        V[b_j*7 + b_k+1][col] += 1
        W[c_k*2 + c_i][col] += -1
        W[c_k*2 + c_i+1][col] += 1
        col += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[a_i*4 + a_j+1][col] += 1
        U[a_i*4 + a_j][col] += -1
        V[b_j*7 + b_k][col] += 1
        V[b_j*7 + b_k+1][col] += 1
        W[c_k*2 + c_i+1][col] += 1
        col += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[a_i*4 + a_j+1][col] += 1
        U[a_i*4 + a_j+2][col] += -1
        V[b_j*7 + b_k+1][col] += 1
        V[b_j*7 + b_k+2][col] += 1
        W[c_k*2 + c_i][col] += 1
        col += 1

    # Helper to add a 2x2x3 block (Rank 11)
    def add_2x2x3(a_i, a_j, b_j, b_k, c_k, c_i):
        nonlocal col
        # A: (a_i, a_j) 2x2
        # B: (b_j, b_k) 2x3
        # C: (c_k, c_i) 2x3
        
        # Part 1: 2x2x2 on first 2 cols of B
        # B cols: b_k, b_k+1
        add_strassen(a_i, a_j, b_j, b_k, c_k, c_i)
        
        # Part 2: 2x2x1 on last col of B
        # B col: b_k+2
        # C col: c_k+2
        
        # N1 = A11 * B13
        U[a_i*4 + a_j][col] += 1
        V[b_j*7 + b_k+2][col] += 1
        W[c_k*2 + c_i+2][col] += 1
        col += 1
        
        # N2 = A12 * B23
        U[a_i*4 + a_j+1][col] += 1
        V[b_j*7 + b_k+3][col] += 1 # Wait, B is 2x3. b_j is row of B.
        # B indices: j in {b_j, b_j+1}, k in {b_k, b_k+1, b_k+2}
        # B13 is row b_j, col b_k+2.
        # B23 is row b_j+1, col b_k+2.
        # Correction:
        # N1 = A11 * B13
        # A11 is a_i, a_j.
        # B13 is b_j, b_k+2.
        # N2 = A12 * B23
        # A12 is a_i, a_j+1.
        # B23 is b_j+1, b_k+2.
        
        # Let's re-verify indices for 2x2x1
        # C13 = A11 B13 + A12 B23
        # C23 = A21 B13 + A22 B23
        
        # N1 = A11 B13
        U[a_i*4 + a_j][col] += 1
        V[b_j*7 + b_k+2][col] += 1
        W[c_k*2 + c_i+2][col] += 1
        col += 1
        
        # N2 = A12 B23
        U[a_i*4 + a_j+1][col] += 1
        V[b_j*7 + b_k+3][col] += 1 # b_j+1 row, b_k+2 col -> (b_j+1)*7 + b_k+2
        # My V index calculation: V[row][col].
        # V index is j*p + k.
        # For B23: j = b_j+1, k = b_k+2.
        # So V[(b_j+1)*7 + b_k+2][col] += 1
        # I wrote V[b_j*7 + b_k+3]. This is wrong.
        
        # Let's rewrite the helper carefully.

    # Reset col
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen(a_i, a_j, b_j, b_k, c_k, c_i):
        nonlocal col
        # A: a_i, a_j (2x2)
        # B: b_j, b_k (2x2)
        # C: c_k, c_i (2x2)
        
        # M1
        U[a_i*4 + a_j][col] += 1
        U[a_i*4 + a_j+1][col] += 1
        V[b_j*7 + b_k][col] += 1
        V[b_j*7 + b_k+1][col] += 1
        W[c_k*2 + c_i][col] += 1
        W[c_k*2 + c_i+1][col] += 1
        col += 1
        
        # M2
        U[a_i*4 + a_j+1][col] += 1
        U[a_i*4 + a_j+2][col] += 1
        V[b_j*7 + b_k][col] += 1
        W[c_k*2 + c_i+1][col] += 1
        col += 1
        
        # M3
        U[a_i*4 + a_j][col] += 1
        V[b_j*7 + b_k+1][col] += 1
        V[b_j*7 + b_k+2][col] += -1
        W[c_k*2 + c_i+1][col] += 1
        col += 1
        
        # M4
        U[a_i*4 + a_j+1][col] += 1
        V[b_j*7 + b_k+1][col] += 1
        V[b_j*7 + b_k][col] += -1
        W[c_k*2 + c_i][col] += 1
        col += 1
        
        # M5
        U[a_i*4 + a_j][col] += 1
        U[a_i*4 + a_j+1][col] += 1
        V[b_j*7 + b_k+1][col] += 1
        W[c_k*2 + c_i][col] += -1
        W[c_k*2 + c_i+1][col] += 1
        col += 1
        
        # M6
        U[a_i*4 + a_j+1][col] += 1
        U[a_i*4 + a_j][col] += -1
        V[b_j*7 + b_k][col] += 1
        V[b_j*7 + b_k+1][col] += 1
        W[c_k*2 + c_i+1][col] += 1
        col += 1
        
        # M7
        U[a_i*4 + a_j+1][col] += 1
        U[a_i*4 + a_j+2][col] += -1
        V[b_j*7 + b_k+1][col] += 1
        V[b_j*7 + b_k+2][col] += 1
        W[c_k*2 + c_i][col] += 1
        col += 1

    def add_2x2x3(a_i, a_j, b_j, b_k, c_k, c_i):
        nonlocal col
        # A: a_i, a_j (2x2)
        # B: b_j, b_k (2x3)
        # C: c_k, c_i (2x3)
        
        # Strassen on first 2 cols
        add_strassen(a_i, a_j, b_j, b_k, c_k, c_i)
        
        # 2x2x1 on last col
        # B col index: b_k+2
        # C col index: c_k+2
        
        # N1 = A11 B13
        U[a_i*4 + a_j][col] += 1
        V[b_j*7 + b_k+2][col] += 1
        W[c_k*2 + c_i+2][col] += 1
        col += 1
        
        # N2 = A12 B23
        U[a_i*4 + a_j+1][col] += 1
        V[(b_j+1)*7 + b_k+2][col] += 1
        W[c_k*2 + c_i+2][col] += 1
        col += 1
        
        # N3 = A21 B13
        U[a_i*4 + a_j+2][col] += 1
        V[b_j*7 + b_k+2][col] += 1
        W[c_k*2 + c_i+2+1][col] += 1
        col += 1
        
        # N4 = A22 B23
        U[a_i*4 + a_j+3][col] += 1
        V[(b_j+1)*7 + b_k+2][col] += 1
        W[c_k*2 + c_i+2+1][col] += 1
        col += 1

    # Block 1: 2x4x4
    # A_L (j=0,1), B_T (j=0,1), C (k=0,1)
    add_strassen(0, 0, 0, 0, 0, 0)
    # A_L (j=0,1), B_T (j=0,1), C (k=2,3)
    add_strassen(0, 0, 0, 2, 2, 0)
    # A_R (j=2,3), B_B (j=2,3), C (k=0,1)
    add_strassen(0, 2, 2, 0, 0, 0)
    # A_R (j=2,3), B_B (j=2,3), C (k=2,3)
    add_strassen(0, 2, 2, 2, 2, 0)
    
    # Block 2: 2x4x3
    # A_L (j=0,1), B_T (j=0,1), C (k=4,5,6)
    add_2x2x3(0, 0, 0, 4, 4, 0)
    # A_R (j=2,3), B_B (j=2,3), C (k=4,5,6)
    add_2x2x3(0, 2, 2, 4, 4, 0)
    
    return {"U": U, "V": V, "W": W}
