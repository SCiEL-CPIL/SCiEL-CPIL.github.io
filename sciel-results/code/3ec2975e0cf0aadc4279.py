def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    # Rank is 35 (5 blocks of 2x2 multiplication, each rank 7)
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U indices: 0:A00, 1:A01, 2:A10, 3:A11
    # V indices: 0..9: B0, 10..19: B1
    # W indices: k*n+i. k=0..9, i=0..1.
    # Block r uses B cols 2r, 2r+1 and C cols 2r, 2r+1.
    
    for r in range(5):
        # V indices for this block
        v_b00 = 2 * r
        v_b01 = 2 * r + 1
        v_b10 = 10 + 2 * r
        v_b11 = 10 + 2 * r + 1
        
        # W indices for this block
        # C[i, k] -> index k*n + i
        w_c00 = (2 * r) * n + 0
        w_c10 = (2 * r) * n + 1
        w_c01 = (2 * r + 1) * n + 0
        w_c11 = (2 * r + 1) * n + 1
        
        # Strassen steps
        # M1 = (A00+A11)(B00+B11)
        col = r * 7 + 0
        U[0][col] += 1
        U[3][col] += 1
        V[v_b00][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1
        W[w_c11][col] += 1
        
        # M2 = (A10+A11)B00
        col = r * 7 + 1
        U[2][col] += 1
        U[3][col] += 1
        V[v_b00][col] += 1
        W[w_c10][col] += 1
        W[w_c11][col] += -1
        
        # M3 = A00(B10-B11)
        col = r * 7 + 2
        U[0][col] += 1
        V[v_b10][col] += 1
        V[v_b11][col] += -1
        W[w_c01][col] += 1
        W[w_c11][col] += 1
        
        # M4 = A11(B01-B00)
        col = r * 7 + 3
        U[3][col] += 1
        V[v_b01][col] += 1
        V[v_b00][col] += -1
        W[w_c00][col] += 1
        W[w_c10][col] += 1
        
        # M5 = (A00+A01)B11
        col = r * 7 + 4
        U[0][col] += 1
        U[1][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += -1
        W[w_c01][col] += 1
        
        # M6 = (A10-A00)(B00+B01)
        col = r * 7 + 5
        U[2][col] += 1
        U[0][col] += -1
        V[v_b00][col] += 1
        V[v_b01][col] += 1
        W[w_c11][col] += 1
        
        # M7 = (A01-A11)(B10+B11)
        col = r * 7 + 6
        U[1][col] += 1
        U[3][col] += -1
        V[v_b10][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1

    return {"U": U, "V": V, "W": W}
