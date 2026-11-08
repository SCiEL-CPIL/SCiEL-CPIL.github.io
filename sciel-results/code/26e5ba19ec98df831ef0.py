"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    import numpy as np
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    # C[i,k] = sum_r U[i,r] * V[k,r] must equal A[i,k]*B[k,k] -> U[i,r]*V[k,r]
    # W[r] = sum_{i,k} U[i,r]*V[k,r] * e_{i,k}
    for r in range(Wa.shape[1]):
        u = Ua[:, r]
        v = Va[:, r]
        w = Wa[:, r]
        # outer product of u (n*m) and v (m*p)
        u2 = u.reshape(n, m)
        v2 = v.reshape(m, p)
        prod = u2 @ v2  # n x p
        w2 = w.reshape(p, n)
        # w2[i,k] should equal prod[k,i]
        for i in range(n):
            for k in range(p):
                if w2[i, k] != prod[k, i]:
                    return False
    return True


def _build_decomposition():
    """
    Recursive block decomposition for 3x3 @ 3x7.
    Split rows of B into 2x4 (Strassen 2x2 core) and 1x3 (schoolbook).
    Total rank = 7 + 3 = 10.
    """
    n, m, p = 3, 3, 7

    # Strassen 2x2 algorithm for the 2x4 block (rows 0,1 of B)
    # A_2x2 = A[0:2, 0:2], B_2x2 = B[0:2, 0:2] (2x2 core)
    # Strassen products M1..M7:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    U = []
    V = []
    W = []

    # --- Strassen block for rows 0,1 of B (2x4) ---
    # We need to compute C[0:2, 0:4] = A[0:2, :] @ B[0:2, :]
    # But A is 3x3, so A[0:2,:] is 2x3. B[0:2,:] is 2x7.
    # Split A[0:2,:] into A_left (2x2) and A_right (2x1)
    # Split B[0:2,:] into B_left (2x4) and B_right (2x3)
    # C[0:2, 0:4] = A_left @ B_left + A_right @ B_right[:, 0:4]
    # Hmm, this is getting complex. Let me use a simpler approach.

    # Actually, let's use the full 3x3 @ 3x7 with a different strategy.
    # Use Strassen on the 2x2 block of A and B, then handle the rest.

    # Let me restart with a cleaner approach:
    # A is 3x3, B is 3x7.
    # Split A into A_top (2x3) and A_bottom (1x3)
    # Split B into B_top (2x7) and B_bottom (1x7)
    # C = A_top @ B_top + A_bottom @ B_bottom

    # For A_top @ B_top (2x3 @ 3x7):
    #   Split B_top into B_top_left (2x4) and B_top_right (2x3)
    #   A_top @ B_top_left: 2x3 @ 2x4... no, B_top is 2x7, so B_top_left is 2x4
    #   Wait, A_top is 2x3, B_top is 2x7. The multiplication is (2x3) @ (3x7) but B_top only has 2 rows.
    #   I'm confusing myself. Let me be very explicit.

    # A: 3 rows, 3 cols. B: 3 rows, 7 cols. C = A @ B: 3x7.
    # C[i, k] = sum_j A[i,j] * B[j,k]

    # Split by rows of B (which are columns of the sum):
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:3] @ B[2:3, :]
    # First term: (3x2) @ (2x7) = 3x7
    # Second term: (3x1) @ (1x7) = 3x7

    # For the second term (schoolbook, rank 3):
    # C[i,k] += A[i,2] * B[2,k]
    # U[i, r] = A[i,2] -> U[i, r] = e_i (1 if row i)
    # V[j, r] = B[j,k] -> V[2, r] = 1, V[0,r]=V[1,r]=0
    # W[k, r] = C contribution -> W[k*n+i, r] = 1

    # For the first term: (3x2) @ (2x7)
    # Split A[:, 0:2] into A_left (3x1) and A_right (3x1) [cols 0 and 1]
    # C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is schoolbook with rank 2*7 = 14... but we can do better.

    # Let's use Strassen on the 2x2 core where possible.
    # Actually, for 3x2 @ 2x7, the natural decomposition is:
    # Split the 2x7 into a 2x2 block and a 2x5 block? No.

    # Let me just do: for the (3x2)@(2x7) part, use the fact that
    # we can think of it as summing over 2 indices. The minimal rank
    # for (n x m) @ (m x p) with m=2 is at most n*p but can be reduced.

    # Actually, let me just use a known result: for 3x3 @ 3x7,
    # a good decomposition uses rank around 20-25.

    # Let me try: schoolbook for the 1x3 part (rank 3),
    # and for the 2x3 @ 3x7 part, use Strassen-like on 2x2 blocks.

    # Simpler: just do full schoolbook but with a small optimization.
    # Schoolbook rank = 3*3*7 = 63.

    # Let me try a block approach:
    # C = A @ B where A is 3x3, B is 3x7.
    # Write A = [A1 | A2] where A1 is 3x2, A2 is 3x1
    # Write B = [B1; B2] where B1 is 2x7, B2 is 1x7
    # C = A1 @ B1 + A2 @ B2

    # A2 @ B2: rank 3 (schoolbook for 3x1 @ 1x7)
    # A1 @ B1: 3x2 @ 2x7. Split B1 = [B1a | B1b] where B1a is 2x4, B1b is 2x3
    # A1 @ B1a: 3x2 @ 2x4. Split A1 = [A1a; A1b] where A1a is 2x2, A1b is 1x2
    # A1a @ B1a: 2x2 @ 2x4. Use Strassen on the 2x2 part of B1a.
    #   B1a = [B1a_1 | B1a_2] where B1a_1 is 2x2, B1a_2 is 2x2
    #   A1a @ B1a_1: 2x2 @ 2x2, use Strassen (rank 7)
    #   A1a @ B1a_2: 2x2 @ 2x2, use Strassen (rank 7)
    #   Total for A1a @ B1a: 14
    # A1b @ B1a: 1x2 @ 2x4, schoolbook rank 4
    # A1a @ B1b: 2x2 @ 2x3, split B1b into 2x2 + 2x1
    #   A1a @ B1b_1: 2x2 @ 2x2, Strassen rank 7
    #   A1a @ B1b_2: 2x2 @ 2x1, schoolbook rank 2
    #   Total: 9
    # A1b @ B1b: 1x2 @ 2x3, schoolbook rank 3

    # Total: 3 + 14 + 4 + 9 + 3 = 33

    # This is better than 63! Let me implement this.

    # Actually, let me simplify. I'll use a helper for 2x2 @ 2x2 Strassen
    # and build up from there.

    U = []
    V = []
    W = []

    def add_term(u_vals, v_vals, w_vals):
        """Add a single rank-1 term. u_vals: dict or list for U row indices,
        v_vals: for V, w_vals: for W."""
        U.append(u_vals)
        V.append(v_vals)
        W.append(w_vals)

    # We'll build terms as lists of (index, value) for each of U, V, W.
    # U has 3*3=9 rows, V has 3*7=21 rows, W has 7*3=21 rows.

    def make_U(indices):
        """indices: list of (row, value)"""
        u = [0.0] * 9
        for idx, val in indices:
            u[idx] = val
        return u

    def make_V(indices):
        v = [0.0] * 21
        for idx, val in indices:
            v[idx] = val
        return v

    def make_W(indices):
        w = [0.0] * 21
        for idx, val in indices:
            w[idx] = val
        return w

    # Strassen for 2x2 @ 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b21-b22)
    # M4 = a22(b12-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    def strassen_2x2(U, V, W, a_rows, b_rows, c_rows, a_cols, b_cols, c_cols):
        """
        a_rows: list of 2 U-row indices for A's rows (0,1)
        b_rows: list of 2 V-row indices for B's rows (0,1)
        c_rows: list of 2 W-row indices for C's rows (0,1)
        a_cols: list of 2 U-row indices for A's cols (0,1) -- wait, U is indexed by (i,j) for A[i,j]
        """
        # U index: i*m + j where i is row, j is col of A
        # V index: j*p + k where j is row of B, k is col of B
        # W index: k*n + i where k is row of C (col of B), i is col of C (row of A)

        # For a 2x2 @ 2x2 subproblem:
        # A submatrix at rows a_rows, cols a_cols
        # B submatrix at rows b_rows, cols b_cols
        # C submatrix at rows c_rows (cols of B), cols c_rows (rows of A)

        ar0, ar1 = a_rows
        ac0, ac1 = a_cols
        br0, br1 = b_rows
        bc0, bc1 = b_cols
        cr0, cr1 = c_rows
        cc0, cc1 = c_rows  # same as c_rows for the column index in W

        # a11 = A[ar0, ac0], a12 = A[ar0, ac1]
        # a21 = A[ar1, ac0], a22 = A[ar1, ac1]
        # b11 = B[br0, bc0], b12 = B[br0, bc1]
        # b21 = B[br1, bc0], b22 = B[br1, bc1]
        # c11 = C[cr0, cc0], c12 = C[cr0, cc1]
        # c21 = C[cr1, cc0], c22 = C[cr1, cc1]

        # U indices:
        u_a11 = ar0 * 3 + ac0
        u_a12 = ar0 * 3 + ac1
        u_a21 = ar1 * 3 + ac0
        u_a22 = ar1 * 3 + ac1

        # V indices:
        v_b11 = br0 * 7 + bc0
        v_b12 = br0 * 7 + bc1
        v_b21 = br1 * 7 + bc0
        v_b22 = br1 * 7 + bc1

        # W indices:
        w_c11 = cc0 * 3 + cr0
        w_c12 = cc1 * 3 + cr0
        w_c21 = cc0 * 3 + cr1
        w_c22 = cc1 * 3 + cr1

        # M1 = (a11+a22)(b11+b22)
        U.append(make_U([(u_a11, 1), (u_a22, 1)]))
        V.append(make_V([(v_b11, 1), (v_b22, 1)]))
        W.append(make_W([(w_c11, 1), (w_c22, 1)]))

        # M2 = (a21+a22)b11
        U.append(make_U([(u_a21, 1), (u_a22, 1)]))
        V.append(make_V([(v_b11, 1)]))
        W.append(make_W([(w_c21, 1), (w_c22, -1)]))

        # M3 = a11(b21-b22)
        U.append(make_U([(u_a11, 1)]))
        V.append(make_V([(v_b21, 1), (v_b22, -1)]))
        W.append(make_W([(w_c12, 1), (w_c22, 1)]))

        # M4 = a22(b12-b11)
        U.append(make_U([(u_a22, 1)]))
        V.append(make_V([(v_b12, 1), (v_b11, -1)]))
        W.append(make_W([(w_c11, 1), (w_c21, 1)]))

        # M5 = (a11+a12)b22
        U.append(make_U([(u_a11, 1), (u_a12, 1)]))
        V.append(make_V([(v_b22, 1)]))
        W.append(make_W([(w_c11, -1), (w_c12, 1)]))

        # M6 = (a21-a11)(b11+b12)
        U.append(make_U([(u_a21, 1), (u_a11, -1)]))
        V.append(make_V([(v_b11, 1), (v_b12, 1)]))
        W.append(make_W([(w_c22, 1)]))

        # M7 = (a12-a22)(b21+b22)
        U.append(make_U([(u_a12, 1), (u_a22, -1)]))
        V.append(make_V([(v_b21, 1), (v_b22, 1)]))
        W.append(make_W([(w_c11, 1), (w_c22, 1)]))

    # Now build the full decomposition:
    # C = A @ B
    # A is 3x3, B is 3x7.
    # Split B into B_top (rows 0,1) and B_bottom (row 2)
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:3] @ B[2:3, :]

    # Part 2: A[:, 2] @ B[2, :] (3x1 @ 1x7), schoolbook rank 3
    for i in range(3):
        for k in range(7):
            U.append(make_U([(i * 3 + 2, 1)]))
            V.append(make_V([(2 * 7 + k, 1)]))
            W.append(make_W([(k * 3 + i, 1)]))

    # Part 1: A[:, 0:2] @ B[0:2, :] (3x2 @ 2x7)
    # Split A[:, 0:2] into A_top (rows 0,1) and A_bottom (row 2)
    # Split B[0:2, :] into B_left (cols 0-3) and B_right (cols 4-6)
    # A_top @ B_left: 2x2 @ 2x4
    #   Split B_left into B_left1 (cols 0,1) and B_left2 (cols 2,3)
    #   A_top @ B_left1: 2x2 @ 2x2, Strassen rank 7
    #   A_top @ B_left2: 2x2 @ 2x2, Strassen rank 7
    # A_bottom @ B_left: 1x2 @ 2x4, schoolbook rank 4
    # A_top @ B_right: 2x2 @ 2x3
    #   Split B_right into B_right1 (cols 4,5) and B_right2 (col 6)
    #   A_top @ B_right1: 2x2 @ 2x2, Strassen rank 7
    #   A_top @ B_right2: 2x2 @ 2x1, schoolbook rank 2
    # A_bottom @ B_right: 1x2 @ 2x3, schoolbook rank 3

    # A_top @ B_left1: rows 0,1 of A, cols 0,1; B rows 0,1, cols 0,1
    strassen_2x2(U, V, W,
                 a_rows=[0, 1], a_cols=[0, 1],
                 b_rows=[0, 1], b_cols=[0, 1],
                 c_rows=[0, 1])

    # A_top @ B_left2: rows 0,1 of A, cols 0,1; B rows 0,1, cols 2,3
    strassen_2x2(U, V, W,
                 a_rows=[0, 1], a_cols=[0, 1],
                 b_rows=[0, 1], b_cols=[2, 3],
                 c_rows=[0, 1])

    # A_bottom @ B_left: row 2 of A, cols 0,1; B rows 0,1, cols 0-3
    # C[2, k] += A[2,0]*B[0,k] + A[2,1]*B[1,k] for k=0,1,2,3
    for k in range(4):
        U.append(make_U([(2 * 3 + 0, 1)]))
        V.append(make_V([(0 * 7 + k, 1)]))
        W.append(make_W([(k * 3 + 2, 1)]))
        U.append(make_U([(2 * 3 + 1, 1)]))
        V.append(make_V([(1 * 7 + k, 1)]))
        W.append(make_W([(k * 3 + 2, 1)]))

    # A_top @ B_right1: rows 0,1 of A, cols 0,1; B rows 0,1, cols 4,5
    strassen_2x2(U, V, W,
                 a_rows=[0, 1], a_cols=[0, 1],
                 b_rows=[0, 1], b_cols=[4, 5],
                 c_rows=[0, 1])

    # A_top @ B_right2: rows 0,1 of A, cols 0,1; B rows 0,1, col 6
    # C[i, 6] += A[i,0]*B[0,6] + A[i,1]*B[1,6] for i=0,1
    for i in range(2):
        U.append(make_U([(i * 3 + 0, 1)]))
        V.append(make_V([(0 * 7 + 6, 1)]))
        W.append(make_W([(6 * 3 + i, 1)]))
        U.append(make_U([(i * 3 + 1, 1)]))
        V.append(make_V([(1 * 7 + 6, 1)]))
        W.append(make_W([(6 * 3 + i, 1)]))

    # A_bottom @ B_right: row 2 of A, cols 0,1; B rows 0,1, cols 4-6
    for k in range(4, 7):
        U.append(make_U([(2 * 3 + 0, 1)]))
        V.append(make_V([(0 * 7 + k, 1)]))
        W.append(make_W([(k * 3 + 2, 1)]))
        U.append(make_U([(2 * 3 + 1, 1)]))
        V.append(make_V([(1 * 7 + k, 1)]))
        W.append(make_W([(k * 3 + 2, 1)]))

    return U, V, W


def construct():
    start = time.time()
    deadline = start + 120

    n, m, p = DIMENSIONS

    try:
        U, V, W = _build_decomposition()
        rank = len(U)

        # Validate
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
