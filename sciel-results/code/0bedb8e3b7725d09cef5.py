"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Part 1: A1*B1 (10 products)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Strassen helper
    def strassen_2x2(A00, A01, A10, A11, B00, B01, B10, B11, C00, C01, C10, C11, col):
        # M1 = (A00+A11)*(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        # M2 = (A10+A11)*B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C00][col] += -1; W[C10][col] += 1
        col += 1
        # M3 = A00*(B10-B11)
        U[A00][col] += 1
        V[B10][col] += 1; V[B11][col] += -1
        W[C01][col] += 1; W[C11][col] += -1
        col += 1
        # M4 = A11*(B00-B10)
        U[A11][col] += 1
        V[B00][col] += 1; V[B10][col] += -1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        # M5 = (A00+A01)*B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        # M6 = (A10-A00)*(B00+B01)
        U[A10][col] += 1; U[A00][col] += -1
        V[B00][col] += 1; V[B01][col] += 1
        W[C00][col] += 1
        col += 1
        # M7 = (A01-A11)*(B01+B11)
        U[A01][col] += 1; U[A11][col] += -1
        V[B01][col] += 1; V[B11][col] += 1
        W[C11][col] += -1
        col += 1
        return col
    
    # A2 indices: A00=1, A01=2, A10=4, A11=5
    A00, A01, A10, A11 = 1, 2, 4, 5
    
    # Part 2: A2*B2[0:2] (7 products)
    B00, B01, B10, B11 = 5, 6, 10, 11
    C00, C01, C10, C11 = 0, 1, 2, 3
    col = strassen_2x2(A00, A01, A10, A11, B00, B01, B10, B11, C00, C01, C10, C11, col)
    
    # Part 3: A2*B2[2:4] (7 products)
    B00, B01, B10, B11 = 7, 8, 12, 13
    C00, C01, C10, C11 = 4, 5, 6, 7
    col = strassen_2x2(A00, A01, A10, A11, B00, B01, B10, B11, C00, C01, C10, C11, col)
    
    # Part 4: A2*B2[4:5] (4 products)
    B0, B1 = 9, 14
    C0, C1 = 8, 9
    
    U[A00][col] = 1; V[B0][col] = 1; W[C0][col] = 1; col += 1
    U[A01][col] = 1; V[B1][col] = 1; W[C0][col] = 1; col += 1
    U[A10][col] = 1; V[B0][col] = 1; W[C1][col] = 1; col += 1
    U[A11][col] = 1; V[B1][col] = 1; W[C1][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
