"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to add to matrices
    def add_U(i, j, col, val):
        U[i * m + j][col] += val

    def add_V(j, k, col, val):
        V[j * p + k][col] += val

    def add_W(k, i, col, val):
        W[k * n + i][col] += val

    # Block 1: Strassen 2x2x2 (cols 0-6)
    # A_sub = A[:, 0:2], B_sub = B[0:2, 0:2], C_sub = C[:, 0:2]
    # M1 = (a+d)(e+h) -> C11(+1), C22(+1)
    add_U(0, 0, 0, 1); add_U(1, 0, 0, 1)
    add_V(0, 0, 0, 1); add_V(1, 1, 0, 1)
    add_W(0, 0, 0, 1); add_W(1, 1, 0, 1)
    
    # M2 = (c+d)e -> C21(+1), C22(-1)
    add_U(1, 0, 1, 1); add_U(1, 1, 1, 1)
    add_V(0, 0, 1, 1)
    add_W(0, 1, 1, 1); add_W(1, 1, 1, -1)
    
    # M3 = a(f-h) -> C12(+1), C22(+1)
    add_U(0, 0, 2, 1)
    add_V(0, 1, 2, 1); add_V(1, 1, 2, -1)
    add_W(1, 0, 2, 1); add_W(1, 1, 2, 1)
    
    # M4 = d(g-e) -> C11(-1), C12(+1)
    add_U(1, 1, 3, 1)
    add_V(1, 0, 3, 1); add_V(0, 0, 3, -1)
    add_W(0, 0, 3, -1); add_W(1, 0, 3, 1)
    
    # M5 = (a+b)h -> C11(+1), C21(+1)
    add_U(0, 0, 4, 1); add_U(0, 1, 4, 1)
    add_V(1, 1, 4, 1)
    add_W(0, 0, 4, 1); add_W(0, 1, 4, 1)
    
    # M6 = (c-a)(e+f) -> C22(+1)
    add_U(1, 0, 5, 1); add_U(0, 0, 5, -1)
    add_V(0, 0, 5, 1); add_V(0, 1, 5, 1)
    add_W(1, 1, 5, 1)
    
    # M7 = (b-d)(g+h) -> C11(-1), C12(+1)
    add_U(0, 1, 6, 1); add_U(1, 1, 6, -1)
    add_V(1, 0, 6, 1); add_V(1, 1, 6, 1)
    add_W(0, 0, 6, -1); add_W(1, 0, 6, 1)

    # Block 2: Rank 4 (2x1x2) (cols 7-10)
    # A_rem = A[:, 2], B_rem = B[2, 0:2], C_rem = C[:, 0:2]
    # C_ik = A_i2 B_2k
    # r=0: A_02 B_20 -> C_00
    add_U(0, 2, 7, 1); add_V(2, 0, 7, 1); add_W(0, 0, 7, 1)
    # r=1: A_02 B_21 -> C_01
    add_U(0, 2, 8, 1); add_V(2, 1, 8, 1); add_W(1, 0, 8, 1)
    # r=2: A_12 B_20 -> C_10
    add_U(1, 2, 9, 1); add_V(2, 0, 9, 1); add_W(0, 1, 9, 1)
    # r=3: A_12 B_21 -> C_11
    add_U(1, 2, 10, 1); add_V(2, 1, 10, 1); add_W(1, 1, 10, 1)

    # Block 3: Rank 6 (2x3x1) (cols 11-16)
    # A = A, B_R = B[:, 2], C_R = C[:, 2]
    # C_i2 = sum_j A_ij B_j2
    # r=0: A_00 B_02 -> C_02
    add_U(0, 0, 11, 1); add_V(0, 2, 11, 1); add_W(2, 0, 11, 1)
    # r=1: A_01 B_12 -> C_02
    add_U(0, 1, 12, 1); add_V(1, 2, 12, 1); add_W(2, 0, 12, 1)
    # r=2: A_02 B_22 -> C_02
    add_U(0, 2, 13, 1); add_V(2, 2, 13, 1); add_W(2, 0, 13, 1)
    # r=3: A_10 B_02 -> C_12
    add_U(1, 0, 14, 1); add_V(0, 2, 14, 1); add_W(2, 1, 14, 1)
    # r=4: A_11 B_12 -> C_12
    add_U(1, 1, 15, 1); add_V(1, 2, 15, 1); add_W(2, 1, 15, 1)
    # r=5: A_12 B_22 -> C_12
    add_U(1, 2, 16, 1); add_V(2, 2, 16, 1); add_W(2, 1, 16, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
