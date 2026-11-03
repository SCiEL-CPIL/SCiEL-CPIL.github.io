"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 4, 6

    # Build the rank-66 decomposition using Strassen for 2x2*2x2 and direct for 1x2*2x2
    # Block structure:
    # A (3x4): rows [0,1] top 2, [2] bottom 1; cols [0,1] left, [2,3] right
    # B (4x6): rows [0,1] top 2, [2,3] bottom 2; cols [0,1], [2,3], [4,5]
    # C (3x6): rows [0,1] top 2, [2] bottom 1; cols [0,1], [2,3], [4,5]
    
    # For each column group (t0,t1):
    #   Top 2 rows of C: 2x2 = A[0:2,0:2]*B[0:2,t0:t1+1] + A[0:2,2:4]*B[2:4,t0:t1+1]
    #     -> 2 Strassen blocks = 14 terms
    #   Bottom 1 row of C: 1x2 = A[2,0:2]*B[0:2,t0:t1+1] + A[2,2:4]*B[2:4,t0:t1+1]
    #     -> 2 direct blocks = 8 terms
    # Total per col group: 22 terms, 3 col groups = 66 terms

    U = np.zeros((n * m, 66))
    V = np.zeros((m * p, 66))
    W = np.zeros((p * n, 66))

    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 7 Strassen terms for 2x2*2x2 multiplication.
        A block: rows (a_r, a_r+1), cols (a_c, a_c+1)
        B block: rows (b_r, b_r+1), cols (b_c, b_c+1)
        C block: rows (c_r, c_r+1), cols (c_c, c_c+1)
        """
        nonlocal col
        # Index into flat arrays
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        # W uses trace convention: W[k*n + i, col]
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)

        # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
        U[A00, col] += 1; U[A11, col] += 1
        V[B00, col] += 1; V[B11, col] += 1
        W[C00, col] += 1; W[C11, col] += 1
        col += 1

        # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
        U[A10, col] += 1; U[A11, col] += 1
        V[B00, col] += 1
        W[C10, col] += 1; W[C11, col] -= 1
        col += 1

        # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
        U[A00, col] += 1
        V[B01, col] += 1; V[B11, col] -= 1
        W[C01, col] += 1; W[C11, col] += 1
        col += 1

        # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
        U[A11, col] += 1
        V[B10, col] += 1; V[B00, col] -= 1
        W[C00, col] += 1; W[C10, col] += 1
        col += 1

        # M5 = (A00+A01)*B11 -> C00 -= M5, C01 += M5
        U[A00, col] += 1; U[A01, col] += 1
        V[B11, col] += 1
        W[C00, col] -= 1; W[C01, col] += 1
        col += 1

        # M6 = (A11-A01)*(B00+B01) -> C11 += M6
        U[A11, col] += 1; U[A01, col] -= 1
        V[B00, col] += 1; V[B01, col] += 1
        W[C11, col] += 1
        col += 1

        # M7 = (A01-A11)*(B10+B11) -> C00 += M7
        U[A01, col] += 1; U[A11, col] -= 1
        V[B10, col] += 1; V[B11, col] += 1
        W[C00, col] += 1
        col += 1

    def add_direct_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 4 direct terms for 1x2 * 2x2 -> 1x2 multiplication.
        A: row a_r, cols (a_c, a_c+1)
        B: rows (b_r, b_r+1), cols (b_c, b_c+1)
        C: row c_r, cols (c_c, c_c+1)
        """
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r

        # A[r,c0]*B[s0,t0] -> C[r,t0]
        U[A0, col] += 1; V[B00, col] += 1; W[C0, col] += 1
        col += 1
        # A[r,c1]*B[s1,t0] -> C[r,t0]
        U[A1, col] += 1; V[B10, col] += 1; W[C0, col] += 1
        col += 1
        # A[r,c0]*B[s0,t1] -> C[r,t1]
        U[A0, col] += 1; V[B01, col] += 1; W[C1, col] += 1
        col += 1
        # A[r,c1]*B[s1,t1] -> C[r,t1]
        U[A1, col] += 1; V[B11, col] += 1; W[C1, col] += 1
        col += 1

    # Build the 66-term decomposition
    col_groups = [(0, 1), (2, 3), (4, 5)]

    for t0, t1 in col_groups:
        # Top 2 rows: Strassen for A[0:2,0:2]*B[0:2,t0:t1+1]
        add_strassen(0, 0, 0, t0, 0, t0)
        # Top 2 rows: Strassen for A[0:2,2:4]*B[2:4,t0:t1+1]
        add_strassen(0, 2, 2, t0, 0, t0)
        # Bottom 1 row: direct for A[2,0:2]*B[0:2,t0:t1+1]
        add_direct_1x2_2x2(2, 0, 0, t0, 2, t0)
        # Bottom 1 row: direct for A[2,2:4]*B[2:4,t0:t1+1]
        add_direct_1x2_2x2(2, 2, 2, t0, 2, t0)

    rank = 66

    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, r):
        """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(s - 1.0) > max_err:
                        max_err = abs(s - 1.0)
        return max_err

    err = verify(U, V, W, rank)
    if err > 1e-10:
        # Fallback to schoolbook rank 72
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        rank = rank_school
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by finding linearly dependent columns
    # For each candidate column c, check if its tensor contribution
    # can be expressed as a linear combination of the others.
    # If so, we can drop column c.

    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Precompute the "flattened" tensor contributions for efficiency
    # T_flat[c] is a vector of length n*m*p where T_flat[c][i*m*p + j*p + k] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Actually, let's just work with the (i,j,k) triples directly.

    # For a given set of columns, the tensor is sum of outer products.
    # To check if column c is redundant: for all (i,j,k),
    #   U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_{l!=c} alpha_l * U[i*m+j,l]*V[j*p+k,l]*W[k*n+i,l]

    # This is a linear system in alpha_l. But the number of equations is n*m*p = 72
    # and unknowns is rank-1 = 65. So it's overdetermined.

    # More efficient: for each (i,j,k), the equation involves specific entries.
    # Let's build the system only for nonzero entries.

    def try_eliminate_column(c, U_mat, V_mat, W_mat, r):
        """Try to eliminate column c. Returns (success, alpha_vector) if successful."""
        if time.time() > deadline - 3:
            return False, None

        # Build equations: for each (i,j,k)
        # lhs = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # rhs = sum_{l != c} alpha_l * U[i*m+j,l] * V[j*p+k,l] * W[k*n+i,l]

        # Only include equations where at least one side is nonzero
        A_rows = []
        b_vals = []
        indices = []  # track which (i,j,k) for each row

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    # Build the row for rhs
                    row_nonzero = False
                    row = np.zeros(r - 1)
                    idx = 0
                    for l in range(r):
                        if l == c:
                            continue
                        val = U_mat[i*m+j, l] * V_mat[j*p+k, l] * W_mat[k*n+i, l]
                        if abs(val) > 1e-15:
                            row[idx] = val
                            row_nonzero = True
                        idx += 1

                    if abs(lhs) > 1e-15 or row_nonzero:
                        A_rows.append(row)
                        b_vals.append(lhs)
                        indices.append((i, j, k))

        if not A_rows:
            # All zero, column c contributes nothing, can eliminate
            return True, np.zeros(r - 1)

        A = np.array(A_rows)
        b = np.array(b_vals)

        # Solve least squares
        try:
            alpha, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=1e-12)
        except:
            return False, None

        # Check residual
        A_alpha = A @ alpha
        residual = np.max(np.abs(A_alpha - b))

        if residual < 1e-10:
            return True, alpha
        return False, None

    # Try to eliminate each column
    for c in range(66):
        if time.time() > deadline - 5:
            break

        success, alpha = try_eliminate_column(c, U, V, W, rank)
        if success:
            # Remove column c
            keep = [l for l in range(rank) if l != c]
            U_new = U[:, keep].copy()
            V_new = V[:, keep].copy()
            W_new = W[:, keep].copy()
            new_rank = rank - 1

            # Verify
            err_new = verify(U_new, V_new, W_new, new_rank)
            if err_new < 1e-10:
                best_rank = new_rank
                best_U, best_V, best_W = U_new, V_new, W_new
                rank = new_rank
                U, V, W = U_new, V_new, W_new
                # Try to eliminate more columns from the reduced set
                # Restart the elimination process with the new rank
                for c2 in range(new_rank):
                    if time.time() > deadline - 5:
                        break
                    success2, alpha2 = try_eliminate_column(c2, U, V, W, rank)
                    if success2:
                        keep2 = [l for l in range(rank) if l != c2]
                        U_new2 = U[:, keep2].copy()
                        V_new2 = V[:, keep2].copy()
                        W_new2 = W[:, keep2].copy()
                        new_rank2 = rank - 1
                        err_new2 = verify(U_new2, V_new2, W_new2, new_rank2)
                        if err_new2 < 1e-10:
                            best_rank = new_rank2
                            best_U, best_V, best_W = U_new2, V_new2, W_new2
                            rank = new_rank2
                            U, V, W = U_new2, V_new2, W_new2
                break  # Restart the outer loop with the new set

    # Use the best result found
    U_final = best_U
    V_final = best_V
    W_final = best_W
    rank_final = best_rank

    # Convert to lists with integer/half-integer entries
    # The entries should be integers (from Strassen/direct constructions)
    U_out = U_final.tolist()
    V_out = V_final.tolist()
    W_out = W_final.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
