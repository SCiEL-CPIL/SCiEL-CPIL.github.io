import numpy as np

def construct():
    n, m, p = 2, 3, 4
    rank = 15
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to set values
    def set_val(mat, r, c, val):
        mat[r, c] = val

    # Part 1a: j in {0,1}, k in {0,1}. Rank 0-6.
    # M1 (r=0)
    set_val(U, 0, 0, 1)  # A00
    set_val(U, 4, 0, 1)  # A11
    set_val(V, 0, 0, 1)  # B00
    set_val(V, 5, 0, 1)  # B11
    set_val(W, 0, 0, 1)  # C00
    set_val(W, 3, 0, 1)  # C11
    
    # M2 (r=1)
    set_val(U, 3, 1, 1)  # A10
    set_val(U, 4, 1, 1)  # A11
    set_val(V, 0, 1, 1)  # B00
    set_val(W, 1, 1, 1)  # C10
    set_val(W, 3, 1, -1) # C11
    
    # M3 (r=2)
    set_val(U, 0, 2, 1)  # A00
    set_val(V, 1, 2, 1)  # B01
    set_val(V, 5, 2, -1) # B11
    set_val(W, 2, 2, 1)  # C01
    set_val(W, 3, 2, 1)  # C11
    
    # M4 (r=3)
    set_val(U, 4, 3, 1)  # A11
    set_val(V, 4, 3, 1)  # B10
    set_val(V, 0, 3, -1) # B00
    set_val(W, 0, 3, 1)  # C00
    set_val(W, 1, 3, 1)  # C10
    
    # M5 (r=4)
    set_val(U, 0, 4, 1)  # A00
    set_val(U, 1, 4, 1)  # A01
    set_val(V, 5, 4, 1)  # B11
    set_val(W, 0, 4, -1) # C00
    set_val(W, 2, 4, 1)  # C01
    
    # M6 (r=5)
    set_val(U, 3, 5, 1)  # A10
    set_val(U, 0, 5, -1) # A00
    set_val(V, 0, 5, 1)  # B00
    set_val(V, 1, 5, 1)  # B01
    set_val(W, 3, 5, 1)  # C11
    
    # M7 (r=6)
    set_val(U, 1, 6, 1)  # A01
    set_val(U, 4, 6, -1) # A11
    set_val(V, 4, 6, 1)  # B10
    set_val(V, 5, 6, 1)  # B11
    set_val(W, 0, 6, 1)  # C00

    # Part 1b: j in {0,1}, k in {2,3}. Rank 7-13.
    # M1' (r=7)
    set_val(U, 0, 7, 1)  # A00
    set_val(U, 4, 7, 1)  # A11
    set_val(V, 2, 7, 1)  # B02
    set_val(V, 7, 7, 1)  # B13
    set_val(W, 4, 7, 1)  # C02
    set_val(W, 7, 7, 1)  # C13
    
    # M2' (r=8)
    set_val(U, 3, 8, 1)  # A10
    set_val(U, 4, 8, 1)  # A11
    set_val(V, 2, 8, 1)  # B02
    set_val(W, 5, 8, 1)  # C12
    set_val(W, 7, 8, -1) # C13
    
    # M3' (r=9)
    set_val(U, 0, 9, 1)  # A00
    set_val(V, 3, 9, 1)  # B03
    set_val(V, 7, 9, -1) # B13
    set_val(W, 6, 9, 1)  # C03
    set_val(W, 7, 9, 1)  # C13
    
    # M4' (r=10)
    set_val(U, 4, 10, 1) # A11
    set_val(V, 6, 10, 1) # B12
    set_val(V, 2, 10, -1)# B02
    set_val(W, 4, 10, 1) # C02
    set_val(W, 5, 10, 1) # C12
    
    # M5' (r=11)
    set_val(U, 0, 11, 1) # A00
    set_val(U, 1, 11, 1) # A01
    set_val(V, 7, 11, 1) # B13
    set_val(W, 4, 11, -1)# C02
    set_val(W, 6, 11, 1) # C03
    
    # M6' (r=12)
    set_val(U, 3, 12, 1) # A10
    set_val(U, 0, 12, -1)# A00
    set_val(V, 2, 12, 1) # B02
    set_val(V, 3, 12, 1) # B03
    set_val(W, 7, 12, 1) # C13
    
    # M7' (r=13)
    set_val(U, 1, 13, 1) # A01
    set_val(U, 4, 13, -1)# A11
    set_val(V, 6, 13, 1) # B12
    set_val(V, 7, 13, 1) # B13
    set_val(W, 4, 13, 1) # C02

    # Part 2: j=2. Rank 14.
    set_val(U, 2, 14, 1) # A02
    set_val(U, 5, 14, 1) # A12
    set_val(V, 8, 14, 1) # B20
    set_val(V, 9, 14, 1) # B21
    set_val(V, 10, 14, 1)# B22
    set_val(V, 11, 14, 1)# B23
    set_val(W, 0, 14, 1) # C00
    set_val(W, 1, 14, 1) # C10
    set_val(W, 2, 14, 1) # C01
    set_val(W, 3, 14, 1) # C11
    set_val(W, 4, 14, 1) # C02
    set_val(W, 5, 14, 1) # C12
    set_val(W, 6, 14, 1) # C03
    set_val(W, 7, 14, 1) # C13

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
