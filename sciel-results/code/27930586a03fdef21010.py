"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    def add_strassen_2x2x2(i0, i1, j0, j1, k0, k1):
        nonlocal col
        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1
        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1
        c00 = k0 * n + i0
        c01 = k0 * n + i1
        c10 = k1 * n + i0
        c11 = k1 * n + i1
        
        # M1 = (A00+A11)(B00+B11), +C00, +C11
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00, +C10, -C11
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] += -1
        col += 1
        
        # M3 = A00(B10-B11), +C01, +C11
        U[a00][col] += 1
        V[b10][col] += 1; V[b11][col] += -1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        
        # M4 = A11(B01-B00), +C00, +C10
        U[a11][col] += 1
        V[b01][col] += 1; V[b00][col] += -1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11, -C00, +C01
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] += -1; W[c01][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B10), +C11
        U[a10][col] += 1; U[a00][col] += -1
        V[b00][col] += 1; V[b10][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B01+B11), +C00
        U[a01][col] += 1; U[a11][col] += -1
        V[b01][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1)
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen_2x2x2(0, 1, 0, 1, 2, 3)
    
    # Block 3: i in {0,1}, j in {0,1}, k=4 (2x2@2x1)
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    # Block 4: i in {0,1}, j=2, k in {0,1} (2x1@1x2)
    for i in range(2):
        for k in range(2):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 5: i in {0,1}, j=2, k in {2,3} (2x1@1x2)
    for i in range(2):
        for k in range(2, 4):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 6: i in {0,1}, j=2, k=4 (2x1@1x1)
    for i in range(2):
        U[i * m + 2][col] = 1
        V[2 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1
    
    assert col == 28, f"Expected 28, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
