DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set values
    def set_val(mat, row, col, val):
        mat[row][col] += val

    # Indices
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    # Strassen for U_A (A00, A01, A10, A11) and X_B (B00, B01, B10, B11)
    # U_A indices: A00=0, A01=1, A10=3, A11=4
    # X_B indices: B00=0, B01=1, B10=3, B11=4
    
    # M1 = (A00+A11)(B00+B11)
    set_val(U, 0, 0, 1)
    set_val(U, 4, 0, 1)
    set_val(V, 0, 0, 1)
    set_val(V, 4, 0, 1)
    set_val(W, 0, 0, 1) # C00
    set_val(W, 3, 0, 1) # C11
    
    # M2 = (A10+A11)B00
    set_val(U, 3, 1, 1)
    set_val(U, 4, 1, 1)
    set_val(V, 0, 1, 1)
    set_val(W, 1, 1, 1) # C10
    set_val(W, 3, 1, -1) # C11
    
    # M3 = A00(B01-B11)
    set_val(U, 0, 2, 1)
    set_val(V, 1, 2, 1)
    set_val(V, 4, 2, -1)
    set_val(W, 2, 2, 1) # C01
    set_val(W, 3, 2, 1) # C11
    
    # M4 = A11(B10-B00)
    set_val(U, 4, 3, 1)
    set_val(V, 3, 3, 1)
    set_val(V, 0, 3, -1)
    set_val(W, 0, 3, 1) # C00
    set_val(W, 1, 3, 1) # C10
    
    # M5 = (A00+A01)B11
    set_val(U, 0, 4, 1)
    set_val(U, 1, 4, 1)
    set_val(V, 4, 4, 1)
    set_val(W, 0, 4, -1) # C00
    set_val(W, 2, 4, 1) # C01
    
    # M6 = (A10-A00)(B00+B01)
    set_val(U, 3, 5, 1)
    set_val(U, 0, 5, -1)
    set_val(V, 0, 5, 1)
    set_val(V, 1, 5, 1)
    set_val(W, 3, 5, 1) # C11
    
    # M7 = (A01-A11)(B10+B11)
    set_val(U, 1, 6, 1)
    set_val(U, 4, 6, -1)
    set_val(V, 3, 6, 1)
    set_val(V, 4, 6, 1)
    set_val(W, 0, 6, 1) # C00
    
    # V_A Z_B: A02, A12 with B20, B21
    # A02=2, A12=5
    # B20=6, B21=7
    
    # A02 B20
    set_val(U, 2, 7, 1)
    set_val(V, 6, 7, 1)
    set_val(W, 0, 7, 1) # C00
    
    # A02 B21
    set_val(U, 2, 8, 1)
    set_val(V, 7, 8, 1)
    set_val(W, 2, 8, 1) # C01
    
    # A12 B20
    set_val(U, 5, 9, 1)
    set_val(V, 6, 9, 1)
    set_val(W, 1, 9, 1) # C10
    
    # A12 B21
    set_val(U, 5, 10, 1)
    set_val(V, 7, 10, 1)
    set_val(W, 3, 10, 1) # C11
    
    # U_A Y_B: A00, A01, A10, A11 with B02, B12
    # B02=2, B12=5
    
    # A00 B02
    set_val(U, 0, 11, 1)
    set_val(V, 2, 11, 1)
    set_val(W, 4, 11, 1) # C02
    
    # A01 B12
    set_val(U, 1, 12, 1)
    set_val(V, 5, 12, 1)
    set_val(W, 4, 12, 1) # C02
    
    # A10 B02
    set_val(U, 3, 13, 1)
    set_val(V, 2, 13, 1)
    set_val(W, 5, 13, 1) # C12
    
    # A11 B12
    set_val(U, 4, 14, 1)
    set_val(V, 5, 14, 1)
    set_val(W, 5, 14, 1) # C12
    
    # V_A W_B: A02, A12 with B22
    # B22=8
    
    # A02 B22
    set_val(U, 2, 15, 1)
    set_val(V, 8, 15, 1)
    set_val(W, 4, 15, 1) # C02
    
    # A12 B22
    set_val(U, 5, 16, 1)
    set_val(V, 8, 16, 1)
    set_val(W, 5, 16, 1) # C12
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
