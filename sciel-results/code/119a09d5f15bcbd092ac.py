"""Tensor decomposition for 3x4 * 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    _deadline = time.time() + 130
    n, m, p = DIMENSIONS  # 3, 4, 6

    # Build initial factor matrices via block construction
    # Rows {0,1} use 2x2x2 Strassen; row 2 uses 1x2x2 blocks
    # 3 column-blocks of width 2 => 3*(7+7+4) = 60 columns
    U = np.zeros((n * m, 60), dtype=float)
    V = np.zeros((m * p, 60), dtype=float)
    W = np.zeros((p * n, 60), dtype=float)

    col = 0

    def add_strassen2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add a 2x2x2 Strassen block. a_r0,a_r0+1 rows; a_c0,a_c0+1 cols; etc."""
        nonlocal col
        A00 = a_r0 * m + a_c0
        A01 = a_r0 * m + a_c0 + 1
        A10 = (a_r0 + 1) * m + a_c0
        A11 = (a_r0 + 1) * m + a_c0 + 1
        B00 = b_r0 * p + b_c0
        B01 = b_r0 * p + b_c0 + 1
        B10 = (b_r0 + 1) * p + b_c0
        B11 = (b_r0 + 1) * p + b_c0 + 1
        C00 = c_c0 * n + c_r0
        C01 = (c_c0 + 1) * n + c_r0
        C10 = c_c0 * n + (c_r0 + 1)
        C11 = (c_c0 + 1) * n + (c_r0 + 1)

        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        W[C01][col] -= 1; W[C10][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] += 1
        col += 1
        # M3 = A00(B01-B11)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] += 1; W[C01][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c0, b_r0, b_c0, c_r, c_c0):
        """Add a 1x2x2 block: (1x2)A(2x2)B -> (1x2)C, 4 multiplications."""
        nonlocal col
        A0 = a_r * m + a_c0
        A1 = a_r * m + a_c0 + 1
        B00 = b_r0 * p + b_c0
        B01 = b_r0 * p + b_c0 + 1
        B10 = (b_r0 + 1) * p + b_c0
        B11 = (b_r0 + 1) * p + b_c0 + 1
        C0 = c_c0 * n + c_r
        C1 = (c_c0 + 1) * n + c_r

        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1; col += 1
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1; col += 1
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1; col += 1
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1; col += 1

    # Apply blocks: for each column-block t in [0,2,4]
    for t in [0, 2, 4]:
        add_strassen2(0, 0, 0, t, 0, t)
        add_strassen2(0, 2, 2, t, 0, t)
        add_1x2_2x2(1, 0, 0, t, 1, t)
        add_1x2_2x2(1, 2, 2, t, 1, t)
        add_1x2_2x2(2, 0, 0, t, 2, t)
        add_1x2_2x2(2, 2, 2, t, 2, t)

    r0 = col  # should be 60

    # Build the M matrix for fast verification
    # M[i,j] = sum_l U[i,l]*V[j,l]*W[k*n+i,l] where k = j//p
    # i in [0, n*m), j in [0, m*p)
    def build_M(U_mat, V_mat, W_mat, r):
        """Build M matrix of shape (n*m, m*p)."""
        M = np.zeros((n * m, m * p), dtype=float)
        for l in range(r):
            u_col = U_mat[:, l]
            v_col = V_mat[:, l]
            w_col = W_mat[:, l]
            # For each i, j: M[i,j] += u_col[i] * v_col[j] * w_col[j//p * n + i]
            # Vectorize: for each j, w_index = (j//p)*n + i
            # M[i,j] += u_col[i] * v_col[j] * w_col[(j//p)*n + i]
            # Reshape: w_col has shape (p*n,). w_col[(j//p)*n + i] = w_col.reshape(p,n)[j//p, i]
            w_res = w_col.reshape(p, n)  # shape (p, n)
            # For each j in [0, m*p): w_val = w_res[j//p, i]
            # M[i,j] += u_col[i] * v_col[j] * w_res[j//p, i]
            # Let's do it block by block over p
            for k in range(p):
                # j ranges from k*p to (k+1)*p - 1
                j_start = k * p
                j_end = (k + 1) * p
                v_block = v_col[j_start:j_end]  # shape (p,)
                w_val = w_res[k, :]  # shape (n,)
                # M[i, j_start:j_end] += u_col[i] * v_block * w_val[i]
                # = outer(u_col * w_val, v_block)
                M[:, j_start:j_end] += np.outer(u_col * w_val, v_block)
        return M

    M = build_M(U, V, W, r0)

    def is_valid(M_mat):
        return np.allclose(M_mat, 0, atol=1e-9)

    # Verify initial
    if not is_valid(M):
        # Fallback: naive
        r_naive = n * m * p
        U = np.zeros((n * m, r_naive), dtype=float)
        V = np.zeros((m * p, r_naive), dtype=float)
        W = np.zeros((p * n, r_naive), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Phase 1: Nullspace-based column elimination
    # M has shape (n*m, m*p) = (12, 24). We need M=0.
    # Each column l of M is M_l = sum of contributions. If we can find a combination
    # of columns that sums to zero, we can eliminate one.
    # Actually, the constraint is that the full M matrix must be zero.
    # The M matrix is built as sum_l outer(u_col_l * w_val_l, v_block_l) for each k.
    # This is a linear system in the columns.
    # Alternative: try dropping columns one at a time.

    # Phase 2: Single drop search
    def try_drop(U_mat, V_mat, W_mat, M_mat, col_idx, r):
        """Try dropping column col_idx. Returns (U,V,W,M) if valid, else None."""
        if time.time() > _deadline - 5:
            return None
        # Remove column
        U_new = np.delete(U_mat, col_idx, axis=1)
        V_new = np.delete(V_mat, col_idx, axis=1)
        W_new = np.delete(W_mat, col_idx, axis=1)
        # Update M: subtract contribution of column col_idx
        M_new = M_mat.copy()
        u_col = U_mat[:, col_idx]
        v_col = V_mat[:, col_idx]
        w_col = W_mat[:, col_idx]
        w_res = w_col.reshape(p, n)
        for k in range(p):
            j_start = k * p
            j_end = (k + 1) * p
            v_block = v_col[j_start:j_end]
            w_val = w_res[k, :]
            M_new[:, j_start:j_end] -= np.outer(u_col * w_val, v_block)
        if is_valid(M_new):
            return U_new, V_new, W_new, M_new
        return None

    improved = True
    while improved and time.time() < _deadline - 10:
        improved = False
        r = U.shape[1]
        for l in range(r):
            if time.time() > _deadline - 10:
                break
            result = try_drop(U, V, W, M, l, r)
            if result is not None:
                U, V, W, M = result
                improved = True
                break

    # Phase 3: Pair drop search
    if time.time() < _deadline - 10:
        r = U.shape[1]
        for l1 in range(r):
            if time.time() > _deadline - 10:
                break
            for l2 in range(l1 + 1, r):
                if time.time() > _deadline - 10:
                    break
                # Try dropping both l1 and l2
                M_test = M.copy()
                for col_idx in [l1, l2]:
                    u_col = U[:, col_idx]
                    v_col = V[:, col_idx]
                    w_col = W[:, col_idx]
                    w_res = w_col.reshape(p, n)
                    for k in range(p):
                        j_start = k * p
                        j_end = (k + 1) * p
                        v_block = v_col[j_start:j_end]
                        w_val = w_res[k, :]
                        M_test[:, j_start:j_end] -= np.outer(u_col * w_val, v_block)
                if is_valid(M_test):
                    remaining = [i for i in range(r) if i != l1 and i != l2]
                    U = U[:, remaining]
                    V = V[:, remaining]
                    W = W[:, remaining]
                    M = M_test
                    r = U.shape[1]
                    break
            if time.time() > _deadline - 10:
                break

    # Phase 4: Triple drop search
    if time.time() < _deadline - 5:
        r = U.shape[1]
        found = False
        for l1 in range(r):
            if time.time() > _deadline - 5 or found:
                break
            for l2 in range(l1 + 1, r):
                if time.time() > _deadline - 5 or found:
                    break
                for l3 in range(l2 + 1, r):
                    if time.time() > _deadline - 5 or found:
                        break
                    M_test = M.copy()
                    for col_idx in [l1, l2, l3]:
                        u_col = U[:, col_idx]
                        v_col = V[:, col_idx]
                        w_col = W[:, col_idx]
                        w_res = w_col.reshape(p, n)
                        for k in range(p):
                            j_start = k * p
                            j_end = (k + 1) * p
                            v_block = v_col[j_start:j_end]
                            w_val = w_res[k, :]
                            M_test[:, j_start:j_end] -= np.outer(u_col * w_val, v_block)
                    if is_valid(M_test):
                        remaining = [i for i in range(r) if i not in (l1, l2, l3)]
                        U = U[:, remaining]
                        V = V[:, remaining]
                        W = W[:, remaining]
                        M = M_test
                        found = True
                        break

    # Phase 5: Another round of single drops after pair/triple
    if time.time() < _deadline - 3:
        improved = True
        while improved and time.time() < _deadline - 3:
            improved = False
            r = U.shape[1]
            for l in range(r):
                if time.time() > _deadline - 3:
                    break
                result = try_drop(U, V, W, M, l, r)
                if result is not None:
                    U, V, W, M = result
                    improved = True
                    break

    # Final verification
    if not is_valid(M):
        # Should not happen, but fallback
        U_final = np.zeros((n * m, n * m * p), dtype=float)
        V_final = np.zeros((m * p, n * m * p), dtype=float)
        W_final = np.zeros((p * n, n * m * p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_final[i * m + j, idx] = 1
                    V_final[j * p + k, idx] = 1
                    W_final[k * n + i, idx] = 1
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
