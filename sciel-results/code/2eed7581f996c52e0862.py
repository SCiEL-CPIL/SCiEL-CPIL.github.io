def construct():
    n, m, p = 2, 5, 6
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal col
        # M1
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        # M2
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        col += 1
        # M3
        U[a00][col] += 1
        V[b10][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        # M4
        U[a11][col] += 1
        V[b01][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        # M5
        U[a00][col] += 1; U[a10][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        # M6
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        # M7
        U[a11][col] += 1; U[a10][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] -= 1
        col += 1

    # Block 1: J1={0,1}
    # A indices: 0,1,5,6
    # B cols 0,1: V indices 0,1,6,7
    # C cols 0,1: W indices 0,1,2,3
    add_strassen(0, 1, 5, 6, 0, 1, 6, 7, 0, 1, 2, 3)
    
    # B cols 2,3: V indices 2,3,8,9
    # C cols 2,3: W indices 4,5,6,7
    add_strassen(0, 1, 5, 6, 2, 3, 8, 9, 4, 5, 6, 7)
    
    # B cols 4,5: V indices 4,5,10,11
    # C cols 4,5: W indices 8,9,10,11
    add_strassen(0, 1, 5, 6, 4, 5, 10, 11, 8, 9, 10, 11)
    
    # Block 2: J2b={3,4}
    # A indices: 3,4,8,9
    # B cols 0,1: V indices 18,19,24,25
    # C cols 0,1: W indices 0,1,2,3
    add_strassen(3, 4, 8, 9, 18, 19, 24, 25, 0, 1, 2, 3)
    
    # B cols 2,3: V indices 20,21,26,27
    # C cols 2,3: W indices 4,5,6,7
    add_strassen(3, 4, 8, 9, 20, 21, 26, 27, 4, 5, 6, 7)
    
    # B cols 4,5: V indices 22,23,28,29
    # C cols 4,5: W indices 8,9,10,11
    add_strassen(3, 4, 8, 9, 22, 23, 28, 29, 8, 9, 10, 11)
    
    # Block 3: J2a={2}
    # A indices: 2, 7
    # B row 2: V indices 12..17
    # C cols 0..5: W indices 0..11
    
    for k in range(6):
        # C0k
        U[2][col] = 1
        V[12+k][col] = 1
        W[k][col] = 1
        col += 1
        # C1k
        U[7][col] = 1
        V[12+k][col] = 1
        W[2+k][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
