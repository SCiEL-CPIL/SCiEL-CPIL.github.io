import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a rank 54 decomposition
    # Split m=5 into 2, 2, 1
    # Split p=6 into 2, 2, 2
    # Block 1: A1 (2x2) * B1 (2x6) -> 3 Strassen blocks (2x2x2)
    # Block 2: A2 (2x2) * B2 (2x6) -> 3 Strassen blocks (2x2x2)
    # Block 3: A3 (2x1) * B3 (1x6) -> 3 Schoolbook blocks (2x1x2)
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # i_off, j_off, k_off are the starting indices for i, j, k
    # The block covers i in [i_off, i_off+2), j in [j_off, j_off+2), k in [k_off, k_off+2)
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00, A11 are 1. V: B00, B11 are 1. W: C00, C11 are 1.
        U[(i_off) * m + (j_off)][col] = 1
        U[(i_off + 1) * m + (j_off + 1)][col] = 1
        V[(j_off) * p + (k_off)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        W[(k_off) * n + (i_off)][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[(i_off + 1) * m + (j_off)][col] = 1
        U[(i_off + 1) * m + (j_off + 1)][col] = 1
        V[(j_off) * p + (k_off)][col] = 1
        W[(k_off) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[(i_off) * m + (j_off)][col] = 1
        V[(j_off) * p + (k_off + 1)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = -1
        W[(k_off + 1) * n + (i_off)][col] = 1
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[(i_off + 1) * m + (j_off + 1)][col] = 1
        V[(j_off + 1) * p + (k_off)][col] = 1
        V[(j_off) * p + (k_off)][col] = -1
        W[(k_off) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[(i_off) * m + (j_off)][col] = 1
        U[(i_off) * m + (j_off + 1)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        W[(k_off + 1) * n + (i_off)][col] = 1
        col += 1
        
        # M6 = (A10 - A00) (B00 + B01)
        U[(i_off + 1) * m + (j_off)][col] = 1
        U[(i_off) * m + (j_off)][col] = -1
        V[(j_off) * p + (k_off)][col] = 1
        V[(j_off) * p + (k_off + 1)][col] = 1
        W[(k_off) * n + (i_off + 1)][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # M7 = (A01 - A11) (B10 + B11)
        U[(i_off) * m + (j_off + 1)][col] = 1
        U[(i_off + 1) * m + (j_off + 1)][col] = -1
        V[(j_off + 1) * p + (k_off)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        W[(k_off) * n + (i_off)][col] = 1
        W[(k_off + 1) * n + (i_off)][col] = 1
        col += 1

    # Helper to add a 2x1x2 schoolbook block
    # i_off, j_off, k_off
    def add_2x1x2(i_off, j_off, k_off):
        nonlocal col
        # A is 2x1, B is 1x2
        # C00 = A0 B0
        U[(i_off) * m + (j_off)][col] = 1
        V[(j_off) * p + (k_off)][col] = 1
        W[(k_off) * n + (i_off)][col] = 1
        col += 1
        
        # C01 = A0 B1
        U[(i_off) * m + (j_off)][col] = 1
        V[(j_off) * p + (k_off + 1)][col] = 1
        W[(k_off + 1) * n + (i_off)][col] = 1
        col += 1
        
        # C10 = A1 B0
        U[(i_off + 1) * m + (j_off)][col] = 1
        V[(j_off) * p + (k_off)][col] = 1
        W[(k_off) * n + (i_off + 1)][col] = 1
        col += 1
        
        # C11 = A1 B1
        U[(i_off + 1) * m + (j_off)][col] = 1
        V[(j_off) * p + (k_off + 1)][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1

    # Block 1: A1 (cols 0-1) * B1 (rows 0-1)
    # B1 is 2x6, split into 3 blocks of 2x2
    add_strassen(0, 0, 0) # i:0-1, j:0-1, k:0-1
    add_strassen(0, 0, 2) # i:0-1, j:0-1, k:2-3
    add_strassen(0, 0, 4) # i:0-1, j:0-1, k:4-5

    # Block 2: A2 (cols 2-3) * B2 (rows 2-3)
    # B2 is 2x6, split into 3 blocks of 2x2
    add_strassen(0, 2, 0) # i:0-1, j:2-3, k:0-1
    add_strassen(0, 2, 2) # i:0-1, j:2-3, k:2-3
    add_strassen(0, 2, 4) # i:0-1, j:2-3, k:4-5

    # Block 3: A3 (col 4) * B3 (row 4)
    # B3 is 1x6, split into 3 blocks of 1x2
    add_2x1x2(0, 4, 0) # i:0-1, j:4, k:0-1
    add_2x1x2(0, 4, 2) # i:0-1, j:4, k:2-3
    add_2x1x2(0, 4, 4) # i:0-1, j:4, k:4-5

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
