"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Hybrid decomposition: use Strassen (rank 7) for 2x2x2 sub-problems
    # and schoolbook for the rest.
    #
    # The 3x5 by 5x5 multiplication is split into 9 sub-problems:
    #   C[i,j] = sum over k of A[i,k]*B[k,j]
    #
    # We partition A into rows [0:2, 2:3] and B into columns [0:2, 2:5]
    # and rows [0:2, 2:5].
    #
    # Sub-problems:
    #   C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:5]*B[2:5, 0:2]
    #   C[0:2, 2:5] = A[0:2, 0:2]*B[0:2, 2:5] + A[0:2, 2:5]*B[2:5, 2:5]
    #   C[2:3, 0:2] = A[2:3, 0:2]*B[0:2, 0:2] + A[2:3, 2:5]*B[2:5, 0:2]
    #   C[2:3, 2:5] = A[2:3, 0:2]*B[0:2, 2:5] + A[2:3, 2:5]*B[2:5, 2:5]
    #
    # For 2x2x2 blocks we use Strassen (rank 7 instead of 8).
    # For other blocks we use schoolbook.
    #
    # Let's identify which sub-problems can use Strassen:
    #   2x2x2 blocks need exactly 2x2 A, 2x2 B, producing 2x2 C.
    #
    # A[0:2, 0:2] (2x2) * B[0:2, 0:2] (2x2) -> 2x2 C[0:2, 0:2] part: Strassen
    # A[0:2, 2:4] (2x2) * B[2:4, 0:2] (2x2) -> 2x2 C[0:2, 0:2] part: Strassen
    # A[0:2, 0:2] (2x2) * B[0:2, 2:4] (2x2) -> 2x2 C[0:2, 2:4] part: Strassen
    # A[0:2, 2:4] (2x2) * B[2:4, 2:4] (2x2) -> 2x2 C[0:2, 2:4] part: Strassen
    #
    # Remaining schoolbook parts:
    # A[0:2, 4] (2x1) * B[4, 0:2] (1x2) -> 2x2: schoolbook rank 2
    # A[0:2, 0:2] (2x2) * B[0:2, 4] (2x1) -> 2x1: schoolbook rank 2
    # A[0:2, 4] (2x1) * B[4, 2:4] (1x2) -> 2x2: schoolbook rank 2
    # A[0:2, 2:5] (2x3) * B[2:5, 4] (3x1) -> 2x1: schoolbook rank 3
    # A[2, 0:5] (1x5) * B[0:5, 0:5] (5x5) -> 1x5: schoolbook rank 5
    #
    # Total rank: 4*7 + 2+2+2+3+5 = 28 + 14 = 42

    n, m, p = 3, 5, 5
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 15 x 42
    V = [[0] * rank for _ in range(m * p)]  # 25 x 42
    W = [[0] * rank for _ in range(p * n)]  # 15 x 42

    col = 0

    # Strassen for 2x2x2: computes C = A * B where A is 2x2, B is 2x2
    # Strassen factors:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    def add_strassen(a_rows, a_cols, b_rows, b_cols):
        nonlocal col
        # a_rows: list of 2 row indices in A
        # a_cols: list of 2 col indices in A
        # b_rows: list of 2 row indices in B
        # b_cols: list of 2 col indices in B
        # Output: C[a_rows[0], b_cols[0]], C[a_rows[0], b_cols[1]],
        #         C[a_rows[1], b_cols[0]], C[a_rows[1], b_cols[1]]
        
        a11, a22 = (a_rows[0], a_cols[0]), (a_rows[1], a_cols[1])
        a12, a21 = (a_rows[0], a_cols[1]), (a_rows[1], a_cols[0])
        b11, b22 = (b_rows[0], b_cols[0]), (b_rows[1], b_cols[1])
        b12, b21 = (b_rows[0], b_cols[1]), (b_rows[1], b_cols[0])
        
        def Uidx(r, c):
            return r * m + c
        def Vidx(r, c):
            return r * p + c
        def Widx(k, i):
            return k * n + i
        
        # M1 = (A11+A22)(B11+B22)
        U[Uidx(*a11)][col] = 1
        U[Uidx(*a22)][col] = 1
        V[Vidx(*b11)][col] = 1
        V[Vidx(*b22)][col] = 1
        W[Widx(b_cols[0], a_rows[0])][col] = 1
        W[Widx(b_cols[1], a_rows[0])][col] = 1
        W[Widx(b_cols[0], a_rows[1])][col] = 1
        W[Widx(b_cols[1], a_rows[1])][col] = -1
        col += 1
        
        # M2 = (A21+A22)B11
        U[Uidx(*a21)][col] = 1
        U[Uidx(*a22)][col] = 1
        V[Vidx(*b11)][col] = 1
        W[Widx(b_cols[0], a_rows[1])][col] = 1
        col += 1
        
        # M3 = A11(B21-B22)
        U[Uidx(*a11)][col] = 1
        V[Vidx(*b21)][col] = 1
        V[Vidx(*b22)][col] = -1
        W[Widx(b_cols[1], a_rows[0])][col] = 1
        W[Widx(b_cols[0], a_rows[1])][col] = 1
        col += 1
        
        # M4 = A22(B12-B11)
        U[Uidx(*a22)][col] = 1
        V[Vidx(*b12)][col] = 1
        V[Vidx(*b11)][col] = -1
        W[Widx(b_cols[0], a_rows[0])][col] = -1
        W[Widx(b_cols[0], a_rows[1])][col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[Uidx(*a11)][col] = 1
        U[Uidx(*a12)][col] = 1
        V[Vidx(*b22)][col] = 1
        W[Widx(b_cols[0], a_rows[0])][col] = -1
        W[Widx(b_cols[1], a_rows[0])][col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[Uidx(*a21)][col] = 1
        U[Uidx(*a11)][col] = -1
        V[Vidx(*b11)][col] = 1
        V[Vidx(*b12)][col] = 1
        W[Widx(b_cols[1], a_rows[1])][col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[Uidx(*a12)][col] = 1
        U[Uidx(*a22)][col] = -1
        V[Vidx(*b21)][col] = 1
        V[Vidx(*b22)][col] = 1
        W[Widx(b_cols[0], a_rows[0])][col] = 1
        W[Widx(b_cols[1], a_rows[1])][col] = 1
        col += 1
    
    def add_schoolbook(a_rows, a_cols, b_rows, b_cols):
        nonlocal col
        # General schoolbook: C[i,j] += A[i,k]*B[k,j]
        for i in a_rows:
            for k in a_cols:
                # B row is k, B cols are b_cols
                for j in b_cols:
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1

    # Strassen blocks (2x2x2):
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1])
    
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen([0, 1], [2, 3], [2, 3], [0, 1])
    
    # 3. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3])
    
    # 4. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen([0, 1], [2, 3], [2, 3], [2, 3])
    
    # Schoolbook blocks:
    # 5. A[0:2, 4] * B[4, 0:2] -> C[0:2, 0:2]
    add_schoolbook([0, 1], [4], [4], [0, 1])
    
    # 6. A[0:2, 0:2] * B[0:2, 4] -> C[0:2, 4:5]
    add_schoolbook([0, 1], [0, 1], [0, 1], [4])
    
    # 7. A[0:2, 4] * B[4, 2:4] -> C[0:2, 2:4]
    add_schoolbook([0, 1], [4], [4], [2, 3])
    
    # 8. A[0:2, 2:5] * B[2:5, 4] -> C[0:2, 4:5]
    add_schoolbook([0, 1], [2, 3, 4], [2, 3, 4], [4])
    
    # 9. A[2, 0:5] * B[0:5, 0:5] -> C[2, 0:5]
    add_schoolbook([2], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
