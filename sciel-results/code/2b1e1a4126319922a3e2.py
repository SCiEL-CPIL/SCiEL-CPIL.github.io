"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # Rank 44 decomposition
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    # Part 1: A_1 B_1 (16 terms)
    # A_1 is col 0 of A (rows 0,1) -> U indices 0, 3
    # B_1 is row 0 of B (cols 0..7) -> V indices 0..7
    # C_part is all of C (since A_1 is first col, it multiplies with first row of B)
    # C[i, k] gets contribution from A[i, 0] * B[0, k]
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part 2: A_2 B_2 (28 terms)
    # A_2 is cols 1,2 of A (rows 0,1) -> U indices 1,2 (row 0) and 4,5 (row 1)
    # B_2 is rows 1,2 of B (cols 0..7)
    # Split B_2 into 4 blocks of width 2: t=0,1,2,3
    # B_2,t is 2x2 matrix from B rows 1,2 and cols 2t, 2t+1
    
    # Strassen indices for A_2:
    # A_11_local = A[0, 1] -> U[1]
    # A_12_local = A[0, 2] -> U[2]
    # A_21_local = A[1, 1] -> U[4]
    # A_22_local = A[1, 2] -> U[5]
    
    # Strassen indices for B_2,t:
    # B_11_local = B[1, 2t] -> V[1*8 + 2t]
    # B_12_local = B[1, 2t+1] -> V[1*8 + 2t+1]
    # B_21_local = B[2, 2t] -> V[2*8 + 2t]
    # B_22_local = B[2, 2t+1] -> V[2*8 + 2t+1]
    
    # Strassen indices for C_2,t (result block t):
    # C_11_local = C[0, 2t] -> W[2t*2 + 0]
    # C_12_local = C[0, 2t+1] -> W[2t*2 + 1]
    # C_21_local = C[1, 2t] -> W[(2t+1)*2 + 0]
    # C_22_local = C[1, 2t+1] -> W[(2t+1)*2 + 1]

    for t in range(4):
        c_base = 16 + t * 7
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1 = (A11 + A22)(B11 + B22)
        c = c_base
        U[1][c] += 1
        U[5][c] += 1
        V[1 * p + k0][c] += 1
        V[2 * p + k1][c] += 1
        W[k0 * n + 0][c] += 1 # C00
        W[k1 * n + 1][c] += 1 # C11

        # M2 = (A21 + A22)B11
        c = c_base + 1
        U[4][c] += 1
        U[5][c] += 1
        V[1 * p + k0][c] += 1
        W[k0 * n + 1][c] += 1 # C10

        # M3 = A11(B12 - B22)
        c = c_base + 2
        U[1][c] += 1
        V[1 * p + k1][c] += 1
        V[2 * p + k1][c] -= 1
        W[k1 * n + 0][c] += 1 # C01
        W[k1 * n + 1][c] += 1 # C11

        # M4 = A22(B21 - B11)
        c = c_base + 3
        U[5][c] += 1
        V[2 * p + k0][c] += 1
        V[1 * p + k0][c] -= 1
        W[k0 * n + 0][c] += 1 # C00
        W[k0 * n + 1][c] += 1 # C10

        # M5 = (A11 + A12)B22
        c = c_base + 4
        U[1][c] += 1
        U[2][c] += 1
        V[2 * p + k1][c] += 1
        W[k0 * n + 0][c] -= 1 # C00
        W[k1 * n + 0][c] += 1 # C01

        # M6 = (A21 - A11)(B11 + B12)
        c = c_base + 5
        U[4][c] += 1
        U[1][c] -= 1
        V[1 * p + k0][c] += 1
        V[1 * p + k1][c] += 1
        W[k1 * n + 1][c] += 1 # C11

        # M7 = (A12 - A22)(B21 + B22)
        c = c_base + 6
        U[2][c] += 1
        U[5][c] -= 1
        V[2 * p + k0][c] += 1
        V[2 * p + k1][c] += 1
        W[k0 * n + 0][c] += 1 # C00

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
