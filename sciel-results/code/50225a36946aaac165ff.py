"""Strassen-based rank-25 decomposition for 2x2 times 2x7 matrix multiplication."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 7
    rank = 25
    U = [[0] * rank for _ in range(n * m)]   # shape (4, 25)
    V = [[0] * rank for _ in range(m * p)]   # shape (14, 25)
    W = [[0] * rank for _ in range(p * n)]   # shape (14, 25)

    col = 0

    def add_strassen_block(col_start):
        """Add a Strassen decomposition for a 2x2 block starting at column col_start."""
        nonlocal col
        # m1 = a*(e+f): A[0,0] * (B[0,cs]+B[0,cs+1])
        U[0][col] += 1
        V[0 * p + col_start][col] += 1
        V[0 * p + col_start + 1][col] += 1
        W[col_start * n + 0][col] += 1
        W[col_start * n + 1][col] += 1
        col += 1

        # m2 = (a+b)*f: (A[0,0]+A[0,1]) * B[0,cs+1]
        U[0][col] += 1
        U[1][col] += 1
        V[0 * p + col_start + 1][col] += 1
        W[col_start * n + 1][col] += 1
        W[col_start * n + 3][col] += 1
        col += 1

        # m3 = (c+d)*e: (A[1,0]+A[1,1]) * B[0,cs]
        U[2][col] += 1
        U[3][col] += 1
        V[0 * p + col_start][col] += 1
        W[col_start * n + 0][col] += 1
        W[col_start * n + 2][col] += 1
        col += 1

        # m4 = d*(h-f): A[1,1] * (B[1,cs+1]-B[0,cs+1])
        U[3][col] += 1
        V[1 * p + col_start + 1][col] += 1
        V[0 * p + col_start + 1][col] += -1
        W[col_start * n + 1][col] += 1
        W[col_start * n + 3][col] += 1
        col += 1

        # m5 = (a+b)*(g+h): (A[0,0]+A[0,1]) * (B[1,cs]+B[1,cs+1])
        U[0][col] += 1
        U[1][col] += 1
        V[1 * p + col_start][col] += 1
        V[1 * p + col_start + 1][col] += 1
        W[col_start * n + 0][col] += 1
        W[col_start * n + 1][col] += 1
        W[col_start * n + 2][col] += 1
        W[col_start * n + 3][col] += 1
        col += 1

        # m6 = (c-a)*(e+f): (A[1,0]-A[0,0]) * (B[0,cs]+B[0,cs+1])
        U[2][col] += 1
        U[0][col] += -1
        V[0 * p + col_start][col] += 1
        V[0 * p + col_start + 1][col] += 1
        W[col_start * n + 2][col] += 1
        W[col_start * n + 3][col] += 1
        col += 1

        # m7 = (b-d)*(g+h): (A[0,1]-A[1,1]) * (B[1,cs]+B[1,cs+1])
        U[1][col] += 1
        U[3][col] += -1
        V[1 * p + col_start][col] += 1
        V[1 * p + col_start + 1][col] += 1
        W[col_start * n + 1][col] += 1
        W[col_start * n + 3][col] += 1
        col += 1

    # Three 2x2 blocks: columns 0-1, 2-3, 4-5
    for block in range(3):
        add_strassen_block(block * 2)

    # One 2x1 block: column 6 (schoolbook rank 4)
    # A is 2x2, B_block is 2x1 (column 6 of B)
    # C_block is 2x1
    # Schoolbook: for i in 0..1, j in 0..1, k in 0..0
    for i in range(n):
        for j in range(m):
            for k in range(1):
                U[i * m + j][col] = 1
                V[j * p + 6][col] = 1
                W[6 * n + i][col] = 1
                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
