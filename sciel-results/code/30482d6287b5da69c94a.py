"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 3 blocks of 2x2
    # A is 2x2, B is 2x6 (3 blocks of 2x2)
    # U rows: 0: A00, 1: A01, 2: A10, 3: A11
    # V rows: 0: B00, 1: B01, 2: B02, 3: B03, 4: B04, 5: B05
    #         6: B10, 7: B11, 8: B12, 9: B13, 10: B14, 11: B15
    # W rows: k*n + i
    # k=0: 0: C00, 1: C01
    # k=1: 2: C10, 3: C11
    # k=2: 4: C20, 5: C21
    # k=3: 6: C30, 7: C31
    # k=4: 8: C40, 9: C41
    # k=5: 10: C50, 11: C51
    
    def set_val(mat, row, col, val):
        if mat[row][col] == 0:
            mat[row][col] = val
        else:
            mat[row][col] += val

    for b in range(3):
        # b is 0, 1, 2
        # k indices for B and W: 2*b, 2*b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1: (A00+A11) * (B0k0+B1k1)
        r = b * 7 + 0
        set_val(U, 0, r, 1)
        set_val(U, 3, r, 1)
        set_val(V, 0 * 6 + k0, r, 1)
        set_val(V, 1 * 6 + k1, r, 1)
        set_val(W, k0 * 2 + 0, r, 1)
        set_val(W, k1 * 2 + 1, r, 1)
        
        # M2: (A10+A11) * B0k0
        r = b * 7 + 1
        set_val(U, 2, r, 1)
        set_val(U, 3, r, 1)
        set_val(V, 0 * 6 + k0, r, 1)
        set_val(W, k0 * 2 + 1, r, 1)
        set_val(W, k1 * 2 + 1, r, -1)
        
        # M3: A00 * (B0k1-B1k1)
        r = b * 7 + 2
        set_val(U, 0, r, 1)
        set_val(V, 0 * 6 + k1, r, 1)
        set_val(V, 1 * 6 + k1, r, -1)
        set_val(W, k0 * 2 + 1, r, 1)
        set_val(W, k1 * 2 + 1, r, 1)
        
        # M4: A11 * (B1k0-B0k0)
        r = b * 7 + 3
        set_val(U, 3, r, 1)
        set_val(V, 1 * 6 + k0, r, 1)
        set_val(V, 0 * 6 + k0, r, -1)
        set_val(W, k0 * 2 + 0, r, 1)
        set_val(W, k0 * 2 + 1, r, 1)
        
        # M5: (A00+A01) * B1k1
        r = b * 7 + 4
        set_val(U, 0, r, 1)
        set_val(U, 1, r, 1)
        set_val(V, 1 * 6 + k1, r, 1)
        set_val(W, k0 * 2 + 0, r, -1)
        set_val(W, k0 * 2 + 1, r, 1)
        
        # M6: (A10-A00) * (B0k0+B1k0)
        r = b * 7 + 5
        set_val(U, 2, r, 1)
        set_val(U, 0, r, -1)
        set_val(V, 0 * 6 + k0, r, 1)
        set_val(V, 1 * 6 + k0, r, 1)
        set_val(W, k1 * 2 + 1, r, 1)
        
        # M7: (A01-A11) * (B0k1+B1k1)
        r = b * 7 + 6
        set_val(U, 1, r, 1)
        set_val(U, 3, r, -1)
        set_val(V, 0 * 6 + k1, r, 1)
        set_val(V, 1 * 6 + k1, r, 1)
        set_val(W, k0 * 2 + 0, r, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
