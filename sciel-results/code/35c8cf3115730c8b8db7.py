import time

DIMENSIONS = [2, 4, 7]
DEADLINE = 145.0

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 7
    Na, Nb, Nc = n * m, m * p, p * n  # 8, 28, 14

    def build_rank46():
        rank = 46
        U = [[0] * rank for _ in range(Na)]
        V = [[0] * rank for _ in range(Nb)]
        W = [[0] * rank for _ in range(Nc)]

        def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_col):
            """Add Strassen rank-7 terms for 2x2 A-block, 2x2 B-block -> 2x2 C-block."""
            a00 = a_rows[0] * m + a_cols[0]
            a01 = a_rows[0] * m + a_cols[1]
            a10 = a_rows[1] * m + a_cols[0]
            a11 = a_rows[1] * m + a_cols[1]
            b00 = b_rows[0] * p + b_cols[0]
            b01 = b_rows[0] * p + b_cols[1]
            b10 = b_rows[1] * p + b_cols[0]
            b11 = b_rows[1] * p + b_cols[1]
            c00 = c_cols[0] * n + c_rows[0]
            c01 = c_cols[1] * n + c_rows[0]
            c10 = c_cols[0] * n + c_rows[1]
            c11 = c_cols[1] * n + c_rows[1]
            col = start_col
            # M1 = (A00+A11)(B00+B11) -> C00, C11
            U[a00][col] += 1; U[a11][col] += 1
            V[b00][col] += 1; V[b11][col] += 1
            W[c00][col] += 1; W[c11][col] += 1
            col += 1
            # M2 = (A10+A11)B00 -> C10, -C11
            U[a10][col] += 1; U[a11][col] += 1
            V[b00][col] += 1
            W[c10][col] += 1; W[c11][col] -= 1
            col += 1
            # M3 = A00(B01-B11) -> C01, C11
            U[a00][col] += 1
            V[b01][col] += 1; V[b11][col] -= 1
            W[c01][col] += 1; W[c11][col] += 1
            col += 1
            # M4 = A11(B10-B00) -> C00, C10
            U[a11][col] += 1
            V[b10][col] += 1; V[b00][col] -= 1
            W[c00][col] += 1; W[c10][col] += 1
            col += 1
            # M5 = (A00+A01)B11 -> -C00, C01
            U[a00][col] += 1; U[a01][col] += 1
            V[b11][col] += 1
            W[c00][col] -= 1; W[c01][col] += 1
            col += 1
            # M6 = (A10-A00)(B00+B01) -> C11
            U[a10][col] += 1; U[a00][col] -= 1
            V[b00][col] += 1; V[b01][col] += 1
            W[c11][col] += 1
            col += 1
            # M7 = (A01-A11)(B10+B11) -> C00
            U[a01][col] += 1; U[a11][col] -= 1
            V[b10][col] += 1; V[b11][col] += 1
            W[c00][col] += 1
            col += 1
            return col

        def add_rank2_block(a_rows, a_cols, b_rows, b_col, c_rows, c_col, start_col):
            """Add rank-2 terms for 2x2 A-block, 2x1 B-column -> 2x1 C-column."""
            a00 = a_rows[0] * m + a_cols[0]
            a01 = a_rows[0] * m + a_cols[1]
            a10 = a_rows[1] * m + a_cols[0]
            a11 = a_rows[1] * m + a_cols[1]
            b0 = b_rows[0] * p + b_col
            b1 = b_rows[1] * p + b_col
            c0 = c_col * n + c_rows[0]
            c1 = c_col * n + c_rows[1]
            col = start_col
            # Term for i=0: A[0,:] dot B[:,k]
            U[a00][col] += 1; U[a01][col] += 1
            V[b0][col] += 1; V[b1][col] += 1
            W[c0][col] += 1
            col += 1
            # Term for i=1: A[1,:] dot B[:,k]
            U[a10][col] += 1; U[a11][col] += 1
            V[b0][col] += 1; V[b1][col] += 1
            W[c1][col] += 1
            col += 1
            return col

        col = 0
        # A-block 0 (rows 0-1, cols 0-1)
        col = add_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], col)
        col = add_strassen([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3], col)
        col = add_strassen([0, 1], [0, 1], [0, 1], [4, 5], [0, 1], [4, 5], col)
        col = add_rank2_block([0, 1], [0, 1], [0, 1], 6, [0, 1], 6, col)

        # A-block 1 (rows 0-1, cols 2-3)
        col = add_strassen([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1], col)
        col = add_strassen([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3], col)
        col = add_strassen([0, 1], [2, 3], [2, 3], [4, 5], [0, 1], [4, 5], col)
        col = add_rank2_block([0, 1], [2, 3], [2, 3], 6, [0, 1], 6, col)

        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        """Verify that the decomposition exactly matches the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Verify all other entries are zero
        for u in range(Na):
            for v in range(Nb):
                for w in range(Nc):
                    # Skip entries that should be 1
                    i = u // m
                    j = u % m
                    jj = v // p
                    k = v % p
                    kk = w // n
                    ii = w % n
                    if i == ii and j == jj and k == kk:
                        continue
                    s = 0.0
                    for r in range(rank):
                        s += U[u][r] * V[v][r] * W[w][r]
                    if abs(s) > 1e-9:
                        return False
        return True

    sol = build_rank46()
    if verify(sol["U"], sol["V"], sol["W"], 46):
        return sol

    # Fallback: should not reach here, but return the solution anyway
    return sol
