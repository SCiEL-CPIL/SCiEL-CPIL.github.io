import time

def construct():
    _t0 = time.time()
    deadline = _t0 + 140.0  # safety margin before 150s kill

    n, m, p = 3, 4, 6

    RANK = 66
    U = [[0] * RANK for _ in range(n * m)]  # (12, 66)
    V = [[0] * RANK for _ in range(m * p)]  # (24, 66)
    W = [[0] * RANK for _ in range(p * n)]  # (18, 66)

    col = 0

    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """1x2 * 2x2 block, rank 4 (schoolbook)."""
        nonlocal col
        A0 = _idx_u(a_r, a_c)
        A1 = _idx_u(a_r, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C0 = _idx_w(c_c, c_r)
        C1 = _idx_w(c_c + 1, c_r)

        U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1
        col += 1
        U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1
        col += 1
        U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1
        col += 1
        U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1
        col += 1

    def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """2x2 * 2x2 block, rank 7 (Strassen)."""
        nonlocal col
        A00 = _idx_u(a_r, a_c)
        A01 = _idx_u(a_r, a_c + 1)
        A10 = _idx_u(a_r + 1, a_c)
        A11 = _idx_u(a_r + 1, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C00 = _idx_w(c_c, c_r)
        C01 = _idx_w(c_c + 1, c_r)
        C10 = _idx_w(c_c, c_r + 1)
        C11 = _idx_w(c_c + 1, c_r + 1)

        # M1 = (A00+A11)(B00+B11)
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1

        # M3 = A00(B01-B11)
        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1

        # M4 = A11(B10-B00)
        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1
        col += 1

    # Build the 66-term decomposition
    # Partition: A rows 1+2, cols 2+2; B rows 2+2, cols 2+2+2
    # Row 0 of C: 1x2*2x2 blocks (rank 4 each)
    # Rows 1-2 of C: 2x2*2x2 Strassen blocks (rank 7 each)
    #
    # C[0, 2j:2j+2] = A[0,0:2]*B[0:2, 2j:2j+2] + A[0,2:4]*B[2:4, 2j:2j+2]
    # C[1:3, 2j:2j+2] = A[1:3,0:2]*B[0:2, 2j:2j+2] + A[1:3,2:4]*B[2:4, 2j:2j+2]

    for j in range(3):
        bc = 2 * j
        # Row 0: two 1x2*2x2 blocks
        _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
        _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)
        # Rows 1-2: two 2x2*2x2 Strassen blocks
        _add_strassen(1, 0, 0, bc, 1, 2 * j)
        _add_strassen(1, 2, 2, bc, 1, 2 * j)

    assert col == RANK, f"Expected {RANK} terms, got {col}"

    # Self-verification with a small random test
    if time.time() < deadline - 5:
        import random
        random.seed(42)
        A_test = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B_test = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]

        # Compute C = A * B directly
        C_direct = [[sum(A_test[i][k] * B_test[k][j] for k in range(m)) for j in range(p)] for i in range(n)]

        # Compute C via tensor decomposition
        C_decomp = [[0] * p for _ in range(n)]
        for r in range(RANK):
            # Compute intermediate: for each (i,j) pair in C
            for i in range(n):
                for j in range(p):
                    # sum over k of U[i*m+k][r] * V[k*p+j][r]
                    s = 0
                    for k in range(m):
                        s += U[i * m + k][r] * V[k * p + j][r]
                    C_decomp[i][j] += s * W[j * n + i][r]

        # Verify
        for i in range(n):
            for j in range(p):
                if C_direct[i][j] != C_decomp[i][j]:
                    raise ValueError(f"Decomposition failed at C[{i}][{j}]: {C_direct[i][j]} vs {C_decomp[i][j]}")

    return {"U": U, "V": V, "W": W}
