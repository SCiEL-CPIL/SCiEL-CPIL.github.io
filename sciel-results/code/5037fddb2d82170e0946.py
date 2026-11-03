"""Rewritten program: structured rank-68 decomposition for 3x4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r*4 + a_c
        A01 = a_r*4 + a_c+1
        A10 = (a_r+1)*4 + a_c
        A11 = (a_r+1)*4 + a_c+1
        
        B00 = b_r*6 + b_c
        B01 = b_r*6 + b_c+1
        B10 = (b_r+1)*6 + b_c
        B11 = (b_r+1)*6 + b_c+1
        
        C00 = c_c*3 + c_r
        C01 = (c_c+1)*3 + c_r
        C10 = c_c*3 + (c_r+1)
        C11 = (c_c+1)*3 + (c_r+1)
        
        # M1
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        # M2
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        # M3
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        # M4
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        # M5
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        # M6
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        # M7
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1

    def add_2x2_2x1(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r*4 + a_c
        A01 = a_r*4 + a_c+1
        A10 = (a_r+1)*4 + a_c
        A11 = (a_r+1)*4 + a_c+1
        B0 = b_r*6 + b_c
        B1 = (b_r+1)*6 + b_c
        C0 = c_c*3 + c_r
        C1 = c_c*3 + (c_r+1)
        
        U[A00][col] += 1; V[B0][col] += 1; W[C0][col] += 1; col += 1
        U[A01][col] += 1; V[B1][col] += 1; W[C0][col] += 1; col += 1
        U[A10][col] += 1; V[B0][col] += 1; W[C1][col] += 1; col += 1
        U[A11][col] += 1; V[B1][col] += 1; W[C1][col] += 1; col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r*4 + a_c
        A1 = a_r*4 + a_c+1
        B00 = b_r*6 + b_c
        B01 = b_r*6 + b_c+1
        B10 = (b_r+1)*6 + b_c
        B11 = (b_r+1)*6 + b_c+1
        C0 = c_c*3 + c_r
        C1 = (c_c+1)*3 + c_r
        
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1; col += 1
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1; col += 1
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1; col += 1
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1; col += 1

    def add_1x2_2x1(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r*4 + a_c
        A1 = a_r*4 + a_c+1
        B0 = b_r*6 + b_c
        B1 = (b_r+1)*6 + b_c
        C0 = c_c*3 + c_r
        
        U[A0][col] += 1; V[B0][col] += 1; W[C0][col] += 1; col += 1
        U[A1][col] += 1; V[B1][col] += 1; W[C0][col] += 1; col += 1

    # C[0:2, 0:3]
    add_strassen(0, 0, 0, 0, 0, 0)
    add_2x2_2x1(0, 0, 0, 2, 0, 2)
    add_strassen(0, 2, 2, 0, 0, 0)
    add_2x2_2x1(0, 2, 2, 2, 0, 2)
    
    # C[0:2, 3:6]
    add_strassen(0, 0, 0, 3, 0, 3)
    add_2x2_2x1(0, 0, 0, 5, 0, 5)
    add_strassen(0, 2, 2, 3, 0, 3)
    add_2x2_2x1(0, 2, 2, 5, 0, 5)
    
    # C[2:3, 0:3]
    add_1x2_2x2(2, 0, 0, 0, 2, 0)
    add_1x2_2x1(2, 0, 0, 2, 2, 2)
    add_1x2_2x2(2, 2, 2, 0, 2, 0)
    add_1x2_2x1(2, 2, 2, 2, 2, 2)
    
    # C[2:3, 3:6]
    add_1x2_2x2(2, 0, 0, 3, 2, 3)
    add_1x2_2x1(2, 0, 0, 5, 2, 5)
    add_1x2_2x2(2, 2, 2, 3, 2, 3)
    add_1x2_2x1(2, 2, 2, 5, 2, 5)
    
    assert col == 68
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
