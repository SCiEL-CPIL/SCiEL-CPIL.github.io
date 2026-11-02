"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition for 2x4x5 matrix multiplication.
    # Strategy:
    # Split B (4x5) into B1 (4x2) and B2 (4x3).
    # C = A B1 + A B2.
    # 
    # Part 1: A B1 (2x4 * 4x2 -> 2x2)
    # Split A into A1 (2x2) and A2 (2x2) horizontally.
    # Split B1 into B11 (2x2) and B12 (2x2) vertically.
    # C1 = A1 B11 + A2 B12.
    # Use Strassen (rank 7) for each 2x2x2 block. Total rank 14.
    #
    # Part 2: A B2 (2x4 * 4x3 -> 2x3)
    # Split B2 into B21 (4x1) and B22 (4x2).
    # C2 = A B21 + A B22.
    #
    # Subpart 2a: A B21 (2x4 * 4x1 -> 2x1)
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B21 into B211 (2x1) and B212 (2x1).
    # C21 = A1 B211 + A2 B212.
    # Each 2x2x1 is schoolbook (rank 4). Total rank 8.
    #
    # Subpart 2b: A B22 (2x4 * 4x2 -> 2x2)
    # Split A into A1 (2x2) and A2 (2x2).
    # Split B22 into B221 (2x2) and B222 (2x2).
    # C22 = A1 B221 + A2 B222.
    # Use Strassen (rank 7) for each. Total rank 14.
    #
    # Total Rank = 14 + 8 + 14 = 36.

    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_strassen(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        # U indices for A (2x2)
        u00 = a_row * m + a_col
        u01 = a_row * m + a_col + 1
        u10 = (a_row + 1) * m + a_col
        u11 = (a_row + 1) * m + a_col + 1
        
        # V indices for B (2x2)
        v00 = b_row * p + b_col
        v01 = b_row * p + b_col + 1
        v10 = (b_row + 1) * p + b_col
        v11 = (b_row + 1) * p + b_col + 1
        
        # W indices for C (2x2)
        w00 = c_col * n + c_row
        w01 = (c_col + 1) * n + c_row
        w10 = c_col * n + c_row + 1
        w11 = (c_col + 1) * n + c_row + 1

        # M1 = (a00+a11)(b00+b11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1

        # M2 = (a10+a11)b00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1

        # M3 = a00(b01-b11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] -= 1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1

        # M4 = a11(b10-b00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1

        # M5 = (a00+a01)b11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1

        # M6 = (a10-a00)(b00+b01)
        U[u10][col] += 1; U[u00][col] -= 1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1

        # M7 = (a01-a11)(b10+b11)
        U[u01][col] += 1; U[u11][col] -= 1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    def add_schoolbook_2x2x1(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        # A is 2x2, B is 2x1, C is 2x1
        u00 = a_row * m + a_col
        u01 = a_row * m + a_col + 1
        u10 = (a_row + 1) * m + a_col
        u11 = (a_row + 1) * m + a_col + 1
        
        v0 = b_row * p + b_col
        v1 = (b_row + 1) * p + b_col
        
        w0 = c_col * n + c_row
        w1 = c_col * n + c_row + 1

        # C0 = A00 B0 + A01 B1
        U[u00][col] = 1; V[v0][col] = 1; W[w0][col] = 1; col += 1
        U[u01][col] = 1; V[v1][col] = 1; W[w0][col] = 1; col += 1
        # C1 = A10 B0 + A11 B1
        U[u10][col] = 1; V[v0][col] = 1; W[w1][col] = 1; col += 1
        U[u11][col] = 1; V[v1][col] = 1; W[w1][col] = 1; col += 1

    # Part 1: A B1 (cols 0,1 of B)
    # A1 (cols 0,1) * B11 (rows 0,1, cols 0,1) -> C (cols 0,1)
    add_strassen(0, 0, 0, 0, 0, 0)
    # A2 (cols 2,3) * B12 (rows 2,3, cols 0,1) -> C (cols 0,1)
    add_strassen(0, 2, 2, 0, 0, 0)

    # Part 2a: A B21 (col 2 of B)
    # A1 (cols 0,1) * B211 (rows 0,1, col 2) -> C (col 2)
    add_schoolbook_2x2x1(0, 0, 0, 2, 0, 2)
    # A2 (cols 2,3) * B212 (rows 2,3, col 2) -> C (col 2)
    add_schoolbook_2x2x1(0, 2, 2, 2, 0, 2)

    # Part 2b: A B22 (cols 3,4 of B)
    # A1 (cols 0,1) * B221 (rows 0,1, cols 3,4) -> C (cols 3,4)
    add_strassen(0, 0, 0, 3, 0, 3)
    # A2 (cols 2,3) * B222 (rows 2,3, cols 3,4) -> C (cols 3,4)
    add_strassen(0, 2, 2, 3, 0, 3)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
