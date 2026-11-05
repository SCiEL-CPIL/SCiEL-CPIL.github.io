"""
Rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x4 matrix.

The strategy uses the Kronecker/block structure:
- Split A's rows into top-2 (i=0,1) and bottom-1 (i=2)
- Split B's columns / C's rows into left-2 (k=0,1) and right-2 (k=2,3)
- Split middle dimension j into j=0,1 and j=2,3

For the top-2 rows of A (2x4 * 4x4), use Strassen's rank-7 decomposition for each
2x2 sub-block, giving 4 blocks x 7 = 28.
For the bottom row of A (1x4 * 4x4), use schoolbook rank-4 for each 2-column block,
giving 4 blocks x 4 = 16.

Total: 44, which is less than the schoolbook 48.

We then attempt to reduce the rank further via linear dependency detection on the
flattened rank-1 tensors, and also try a structured search for additional
dependencies using the fact that the 1x2 * 2x2 blocks have a special form.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 118.0

    n, m, p = DIMENSIONS

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def _build_rank7_2x2():
        """Strassen rank-7 decomposition for 2x2 matrix multiplication."""
        terms = []
        terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}))
        terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}))
        terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}))
        terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}))
        terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        return terms

    rank7_terms = _build_rank7_2x2()

    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    def add_rank7(a_row_off, b_col_off, j_off, c_row_off, c_col_off, col_start):
        for r in range(7):
            u_d, v_d, w_d = rank7_terms[r]
            c = col_start + r
            for (ai, aj), val in u_d.items():
                U[(a_row_off + ai) * m + (j_off + aj)][c] += val
            for (bj, bk), val in v_d.items():
                V[(j_off + bj) * p + (b_col_off + bk)][c] += val
            for (ci, ck), val in w_d.items():
                W[(c_col_off + ck) * n + (c_row_off + ci)][c] += val
        return col_start + 7

    def add_rank4(a_row, b_col_off, j_off, c_col_off, col_start):
        for r in range(4):
            c = col_start + r
            j_idx = j_off + (r % 2)
            k_idx = b_col_off + (r // 2)
            U[a_row * m + j_idx][c] = 1
            V[j_idx * p + k_idx][c] = 1
            W[k_idx * n + a_row][c] = 1
        return col_start + 4

    col = 0
    col = add_rank7(0, 0, 0, 0, 0, col)
    col = add_rank7(0, 0, 2, 0, 0, col)
    col = add_rank7(0, 2, 0, 0, 2, col)
    col = add_rank7(0, 2, 2, 0, 2, col)
    col = add_rank4(2, 0, 0, 0, col)
    col = add_rank4(2, 0, 2, 0, col)
    col = add_rank4(2, 2, 0, 2, col)
    col = add_rank4(2, 2, 2, 2, col)

    assert col == total_rank

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Verify the base decomposition
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    C_computed = np.zeros((n, p))
    for r in range(total_rank):
        a_r = np.dot(U_arr[:, r], A_test.flatten())
        b_r = np.dot(V_arr[:, r], B_test.flatten())
        C_computed += W_arr[:, r].reshape(p, n).T * (a_r * b_r)

    if np.max(np.abs(C_computed - C_expected)) > 1e-9:
        return {"U": U_school, "V": V_school, "W": W_school}

    # Attempt rank reduction via linear dependency detection
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    flat_dim = dim_u * dim_v * dim_w

    tensor_vecs = np.zeros((total_rank, flat_dim))
    for r in range(total_rank):
        tensor_vecs[r] = np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r]).flatten()

    removed = set()
    col_order = list(range(28, 44)) + list(range(0, 28))

    for c in col_order:
        if c in removed:
            continue
        if time.time() > deadline - 15:
            break

        remaining = [r for r in range(total_rank) if r not in removed and r != c]
        if not remaining:
            continue

        M = tensor_vecs[remaining]
        try:
            sv = np.linalg.svd(M, compute_uv=False)
            rank_M = int(np.sum(sv > 1e-10))
            M_plus = np.vstack([M, tensor_vecs[c].reshape(1, -1)])
            sv_plus = np.linalg.svd(M_plus, compute_uv=False)
            rank_M_plus = int(np.sum(sv_plus > 1e-10))
        except Exception:
            continue

        if rank_M_plus == rank_M:
            removed.add(c)

    if removed:
        keep = [c for c in range(total_rank) if c not in removed]
        new_rank = len(keep)

        U_new = U_arr[:, keep]
        V_new = V_arr[:, keep]
        W_new = W_arr[:, keep]

        np.random.seed(123)
        A_test2 = np.random.randn(n, m)
        B_test2 = np.random.randn(m, p)
        C_expected2 = A_test2 @ B_test2

        C_computed2 = np.zeros((n, p))
        for r in range(new_rank):
            a_r = np.dot(U_new[:, r], A_test2.flatten())
            b_r = np.dot(V_new[:, r], B_test2.flatten())
            C_computed2 += W_new[:, r].reshape(p, n).T * (a_r * b_r)

        if np.max(np.abs(C_computed2 - C_expected2)) < 1e-9:
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

    # Second phase: try to find structured linear combinations that allow
    # merging of schoolbook columns with Strassen columns.
    # The key observation: for the bottom row (i=2), the schoolbook terms
    # W[k*n+2, col] = 1 for specific (k) values. These are sparse and might
    # be expressible as combinations of the Strassen W vectors.
    # We set up a least-squares problem: for each schoolbook column c_school,
    # find alpha such that tensor_vecs[c_school] ≈ sum alpha_r * tensor_vecs[c_strassen]
    # and then check if the residual is actually zero (within tolerance).

    if time.time() < deadline - 20:
        # Try to express schoolbook columns as combinations of Strassen columns
        # plus other schoolbook columns
        strassen_cols = list(range(28))
        schoolbook_cols = list(range(28, 44))

        for c_s in schoolbook_cols:
            if c_s in removed:
                continue
            if time.time() > deadline - 15:
                break

            # Try to express column c_s in terms of all other non-removed columns
            others = [r for r in range(total_rank) if r != c_s and r not in removed]
            if len(others) < 10:
                continue

            M_others = tensor_vecs[others]  # shape (len(others), flat_dim)
            # Solve M_others.T @ x = tensor_vecs[c_s]
            # Use lstsq
            try:
                x, residuals, rank_x, sv_vals = np.linalg.lstsq(
                    M_others.T, tensor_vecs[c_s], rcond=1e-12
                )
            except Exception:
                continue

            # Check residual
            recon = M_others.T @ x
            res = np.linalg.norm(tensor_vecs[c_s] - recon)
            norm_c = np.linalg.norm(tensor_vecs[c_s])
            if norm_c > 0 and res / norm_c < 1e-10:
                removed.add(c_s)

        if removed:
            keep = [c for c in range(total_rank) if c not in removed]
            new_rank = len(keep)

            U_new = U_arr[:, keep]
            V_new = V_arr[:, keep]
            W_new = W_arr[:, keep]

            np.random.seed(456)
            A_test3 = np.random.randn(n, m)
            B_test3 = np.random.randn(m, p)
            C_expected3 = A_test3 @ B_test3

            C_computed3 = np.zeros((n, p))
            for r in range(new_rank):
                a_r = np.dot(U_new[:, r], A_test3.flatten())
                b_r = np.dot(V_new[:, r], B_test3.flatten())
                C_computed3 += W_new[:, r].reshape(p, n).T * (a_r * b_r)

            if np.max(np.abs(C_computed3 - C_expected3)) < 1e-9:
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
