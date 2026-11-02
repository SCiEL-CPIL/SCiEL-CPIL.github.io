"""Matrix multiplication 2x2x5 rank-19 decomposition."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    assert (n, m, p) == (2, 2, 5)

    # Split B's 5 columns into first 2 (handled by Strassen, rank 7)
    # and remaining 3 (handled by schoolbook, rank 12).
    # Total rank = 7 + 12 = 19.

    # ---- Block 1: Strassen rank-7 for 2x2x2 ----
    # A is 2x2, B1 is 2x2 (columns 0,1 of B)
    # Strassen 7 products for (A * B1):
    #   M1 = (A11 - A22)(B21 + B22)   -> indices in A: (0,0),(1,1); in B1: (1,0),(1,1)
    #   M2 = (A11 + A22) B22          -> A: (0,0),(1,1); B1: (1,1)
    #   M3 = A11 (B11 - B21)          -> A: (0,0); B1: (0,0),(1,0)
    #   M4 = A22 (B12 + B22)          -> A: (1,1); B1: (0,1),(1,1)
    #   M5 = (A11 + A12) B21          -> A: (0,0),(0,1); B1: (1,0)
    #   M6 = (A21 - A11)(B11 + B12)   -> A: (1,0),(0,0); B1: (0,0),(0,1)
    #   M7 = (A12 + A22)(B21 - B11)   -> A: (0,1),(1,1); B1: (1,0),(0,0)
    #
    # C1 = A * B1 (2x2), with:
    #   C111 = M3 + M4 - M2 + M7
    #   C112 = M3 + M4
    #   C121 = M1 + M2
    #   C122 = M1 - M3 + M5 + M6
    #
    # A indices: row = i*m+j = i*2+j  (i in {0,1}, j in {0,1})
    #   (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B1 indices (within block): row = j*p1+k1, p1=2
    #   j=0: k1=0->0, k1=1->1
    #   j=1: k1=0->2, k1=1->3
    # But in the full V, B1 uses global B columns k=0,1
    #   B1[j, k1] maps to global B row j*5 + k1:
    #   j=0,k1=0 -> 0; j=0,k1=1 -> 1
    #   j=1,k1=0 -> 5; j=1,k1=1 -> 6
    #
    # C1 indices: W row = k*n + i, k in {0,1} (global k)
    #   k=0,i=0 -> 0; k=0,i=1 -> 1
    #   k=1,i=0 -> 2; k=1,i=1 -> 3

    strassen_U = [
        [0,  1, 0, 0, 0, -1, 0],  # (0,0)
        [0,  0, 1, 0, 1,  0, 1],  # (0,1)
        [1,  0, 0, 0, 0,  1, 0],  # (1,0)
        [-1, 1, 0, 1, 0,  0, 1],  # (1,1)
    ]
    strassen_V = [
        [0, 0, 1, 0, 1, 1, -1],  # j=0,k1=0 (B1 row 0)
        [0, 0, 0, 1, 0, 1, 0],   # j=0,k1=1 (B1 row 1)
        [1, 1, -1, 0, 0, 0, 1],  # j=1,k1=0 (B1 row 2)
        [1, 1, 0, 1, 0, 0, -1],  # j=1,k1=1 (B1 row 3)
    ]
    strassen_W = [
        [0, 0, 1, 1, 0, 0, 1],  # k=0,i=0
        [0, 0, 0, 1, 0, 0, 0],  # k=0,i=1
        [1, 1, 0, 0, 1, 0, 0],  # k=1,i=0
        [1, -1, 1, 0, 1, 1, 0], # k=1,i=1
    ]

    # ---- Block 2: Schoolbook rank-12 for 2x2x3 ----
    # A is 2x2, B2 is 2x3 (columns 2,3,4 of B)
    # For each (i, j, k2) with i in {0,1}, j in {0,1}, k2 in {0,1,2}:
    #   A[i,j] * B[j, k2] -> C[i, k2]
    # 4*3 = 12 products.
    # A indices: i*2+j
    # B2 global indices: j*5 + (k2+2)
    #   j=0: k2=0->2, k2=1->3, k2=2->4
    #   j=1: k2=0->7, k2=1->8, k2=2->9
    # C2 global W indices: (k2+2)*2 + i
    #   k2=0: i=0->4, i=1->5
    #   k2=1: i=0->6, i=1->7
    #   k2=2: i=0->8, i=1->9

    school_U = []
    school_V = []
    school_W = []
    for i in range(2):
        for j in range(2):
            for k2 in range(3):
                col = len(school_U)
                school_U.append([0] * 12)
                school_V.append([0] * 12)
                school_W.append([0] * 12)
                school_U[i * 2 + j][col] = 1
                school_V[j * 5 + (k2 + 2)][col] = 1
                school_W[(k2 + 2) * 2 + i][col] = 1

    # ---- Combine into full matrices ----
    # U: 4 x 19 (A has 4 entries)
    # V: 10 x 19 (B has 10 entries: j*5+k)
    # W: 10 x 19 (C has 10 entries: k*2+i)

    rank = 19
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(10)]
    W = [[0] * rank for _ in range(10)]

    # Block 1 (Strassen) uses columns 0..6
    for r in range(4):
        for c in range(7):
            U[r][c] = strassen_U[r][c]
    # B1 rows: j=0 -> V rows 0,1; j=1 -> V rows 5,6
    # strassen_V row 0 (j=0,k1=0) -> V[0], row 1 (j=0,k1=1) -> V[1]
    # strassen_V row 2 (j=1,k1=0) -> V[5], row 3 (j=1,k1=1) -> V[6]
    V_row_map = [0, 1, 5, 6]
    for r_local, r_global in enumerate(V_row_map):
        for c in range(7):
            V[r_global][c] = strassen_V[r_local][c]
    # W block 1: k=0,1 -> W rows 0,1,2,3
    for r in range(4):
        for c in range(7):
            W[r][c] = strassen_W[r][c]

    # Block 2 (Schoolbook) uses columns 7..18
    for c in range(12):
        col_global = 7 + c
        for r in range(4):
            U[r][col_global] = school_U[r][c]
        for r in range(10):
            V[r][col_global] = school_V[r][c]
        for r in range(10):
            W[r][col_global] = school_W[r][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
