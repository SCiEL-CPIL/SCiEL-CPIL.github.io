"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set entries
    def set_entry(mat, i, j, val):
        mat[i][j] += val

    # Strassen part for 2x2 submatrix (indices 0,1 in 0-based for A rows, 0,1 for B cols)
    # A indices: 0:A11, 1:A12, 2:A13, 3:A21, 4:A22, 5:A23
    # B indices: 0:B11, 1:B12, 2:B13, 3:B21, 4:B22, 5:B23, 6:B31, 7:B32, 8:B33
    # C indices (W): k*n+i. n=2.
    # C11: k=0, i=0 -> 0
    # C12: k=1, i=0 -> 2
    # C13: k=2, i=0 -> 4
    # C21: k=0, i=1 -> 1
    # C22: k=1, i=1 -> 3
    # C23: k=2, i=1 -> 5

    # Col 1: M1 = (A11+A22)(B11+B22)
    set_entry(U, 0, 0, 1) # A11
    set_entry(U, 4, 0, 1) # A22
    set_entry(V, 0, 0, 1) # B11
    set_entry(V, 4, 0, 1) # B22
    set_entry(W, 0, 0, 1) # C11
    set_entry(W, 3, 0, 1) # C22

    # Col 2: M2 = (A21+A22)B11
    set_entry(U, 3, 1, 1) # A21
    set_entry(U, 4, 1, 1) # A22
    set_entry(V, 0, 1, 1) # B11
    set_entry(W, 1, 1, 1) # C21
    set_entry(W, 3, 1, -1) # C22

    # Col 3: M3 = A11(B12-B22)
    set_entry(U, 0, 2, 1) # A11
    set_entry(V, 1, 2, 1) # B12
    set_entry(V, 4, 2, -1) # B22
    set_entry(W, 2, 2, 1) # C12
    set_entry(W, 3, 2, 1) # C22

    # Col 4: M4 = A22(B21-B11)
    set_entry(U, 4, 3, 1) # A22
    set_entry(V, 3, 3, 1) # B21
    set_entry(V, 0, 3, -1) # B11
    set_entry(W, 0, 3, 1) # C11
    set_entry(W, 1, 3, 1) # C21

    # Col 5: M5 = (A11+A12)B22
    set_entry(U, 0, 4, 1) # A11
    set_entry(U, 1, 4, 1) # A12
    set_entry(V, 4, 4, 1) # B22
    set_entry(W, 0, 4, -1) # C11
    set_entry(W, 2, 4, 1) # C12

    # Col 6: M6 = (A21-A11)(B11+B12)
    set_entry(U, 3, 5, 1) # A21
    set_entry(U, 0, 5, -1) # A11
    set_entry(V, 0, 5, 1) # B11
    set_entry(V, 1, 5, 1) # B12
    set_entry(W, 3, 5, 1) # C22

    # Col 7: M7 = (A12-A22)(B21+B22)
    set_entry(U, 1, 6, 1) # A12
    set_entry(U, 4, 6, -1) # A22
    set_entry(V, 3, 6, 1) # B21
    set_entry(V, 4, 6, 1) # B22
    set_entry(W, 0, 6, 1) # C11

    # Remaining schoolbook terms
    # Col 8: A13 B31 -> C11
    set_entry(U, 2, 7, 1) # A13
    set_entry(V, 6, 7, 1) # B31
    set_entry(W, 0, 7, 1) # C11

    # Col 9: A13 B32 -> C12
    set_entry(U, 2, 8, 1) # A13
    set_entry(V, 7, 8, 1) # B32
    set_entry(W, 2, 8, 1) # C12

    # Col 10: A13 B33 -> C13
    set_entry(U, 2, 9, 1) # A13
    set_entry(V, 8, 9, 1) # B33
    set_entry(W, 4, 9, 1) # C13

    # Col 11: A23 B31 -> C21
    set_entry(U, 5, 10, 1) # A23
    set_entry(V, 6, 10, 1) # B31
    set_entry(W, 1, 10, 1) # C21

    # Col 12: A23 B32 -> C22
    set_entry(U, 5, 11, 1) # A23
    set_entry(V, 7, 11, 1) # B32
    set_entry(W, 3, 11, 1) # C22

    # Col 13: A23 B33 -> C23
    set_entry(U, 5, 12, 1) # A23
    set_entry(V, 8, 12, 1) # B33
    set_entry(W, 5, 12, 1) # C23

    # Col 14: A11 B13 -> C13
    set_entry(U, 0, 13, 1) # A11
    set_entry(V, 2, 13, 1) # B13
    set_entry(W, 4, 13, 1) # C13

    # Col 15: A21 B13 -> C23
    set_entry(U, 3, 14, 1) # A21
    set_entry(V, 2, 14, 1) # B13
    set_entry(W, 5, 14, 1) # C23

    # Col 16: A12 B23 -> C13
    set_entry(U, 1, 15, 1) # A12
    set_entry(V, 5, 15, 1) # B23
    set_entry(W, 4, 15, 1) # C13

    # Col 17: A22 B23 -> C23
    set_entry(U, 4, 16, 1) # A22
    set_entry(V, 5, 16, 1) # B23
    set_entry(W, 5, 16, 1) # C23

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
