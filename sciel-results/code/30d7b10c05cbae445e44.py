import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def time_ok():
        return time.time() < deadline

    def build_structural_decomposition(variant):
        """
        Build a structural decomposition.
        variant 0: row 0 schoolbook, rows 1-2 Strassen
        variant 1: row 2 schoolbook, rows 0-1 Strassen
        variant 2: row 1 schoolbook, rows 0-2 with mixed approach
        Returns list of (u_dict, v_dict, w_dict) tuples with flat indices.
        """
        terms = []

        def add_schoolbook(i, j, k):
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            terms.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))

        def add_strassen(i1, i2, j1, j2, k1, k2):
            """Strassen 2x2x2 block for rows i1,i2, cols j1,j2 of A, rows j1,j2 of B, cols k1,k2 of C"""
            # A indices: (i,j) -> i*m+j
            a = i1 * m + j1
            b = i1 * m + j2
            c = i2 * m + j1
            d = i2 * m + j2
            # B indices: (j,k) -> j*p+k
            e = j1 * p + k1
            f = j1 * p + k2
            g = j2 * p + k1
            h = j2 * p + k2
            # C/W indices: (k,i) -> k*n+i
            p_idx = k1 * n + i1
            q_idx = k2 * n + i1
            r_idx = k1 * n + i2
            s_idx = k2 * n + i2

            # M1 = (A11+A22)(B11+B22)
            terms.append(({a: 1.0, d: 1.0}, {e: 1.0, h: 1.0}, {p_idx: 1.0, s_idx: 1.0}))
            # M2 = (A21+A22)B11
            terms.append(({c: 1.0, d: 1.0}, {e: 1.0}, {r_idx: 1.0, s_idx: -1.0}))
            # M3 = A11(B12-B22)
            terms.append(({a: 1.0}, {f: 1.0, h: -1.0}, {q_idx: 1.0, s_idx: 1.0}))
            # M4 = A22(B21-B11)
            terms.append(({d: 1.0}, {g: 1.0, e: -1.0}, {p_idx: 1.0, r_idx: 1.0}))
            # M5 = (A11+A12)B22
            terms.append(({a: 1.0, b: 1.0}, {h: 1.0}, {p_idx: -1.0, q_idx: 1.0}))
            # M6 = (A21-A11)(B11+B12)
            terms.append(({c: 1.0, a: -1.0}, {e: 1.0, f: 1.0}, {s_idx: 1.0}))
            # M7 = (A12-A22)(B21+B22)
            terms.append(({b: 1.0, d: -1.0}, {g: 1.0, h: 1.0}, {p_idx: 1.0}))

        if variant == 0:
            # Row 0 schoolbook, rows 1-2 Strassen
            for j in range(m):
                for k in range(p):
                    add_schoolbook(0, j, k)
            for j1 in [0, 2]:
                for k1 in [0, 2, 4]:
                    add_strassen(1, 2, j1, j1 + 1, k1, k1 + 1)
        elif variant == 1:
            # Row 2 schoolbook, rows 0-1 Strassen
            for j in range(m):
                for k in range(p):
                    add_schoolbook(2, j, k)
            for j1 in [0, 2]:
                for k1 in [0, 2, 4]:
                    add_strassen(0, 1, j1, j1 + 1, k1, k1 + 1)
        elif variant == 2:
            # Row 1 schoolbook, rows 0 and 2 handled separately
            # Row 0: schoolbook for k=0,1,2; Strassen-like for k=3,4,5 (but can't do 2x2x2 with only 1 row)
            # Actually, let's do: row 0 and row 2 each get schoolbook, row 1 gets schoolbook
            # This is just full schoolbook = 72. Not useful.
            # Instead: try rows 0-1 Strassen for j=0-3, k=0-5, and row 2 schoolbook
            # That's same as variant 1.
            # Let's try: row 0 schoolbook, row 1 schoolbook, row 2 with Strassen on 2x2x2 blocks
            # But row 2 alone can't do Strassen. So this variant is just schoolbook.
            for j in range(m):
                for k in range(p):
                    add_schoolbook(0, j, k)
                    add_schoolbook(1, j, k)
                    add_schoolbook(2, j, k)
        return terms

    def terms_to_matrices(terms):
        """Convert list of term dicts to U, V, W matrices."""
        rank = len(terms)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for idx, val in u_dict.items():
                U[idx, r] = val
            for idx, val in v_dict.items():
                V[idx, r] = val
            for idx, val in w_dict.items():
                W[idx, r] = val

        return U, V, W, rank

    def verify_decomposition(U, V, W):
        """Verify tensor identity: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def verify_fast(U, V, W):
        """Faster verification using numpy."""
        rank = U.shape[1]
        # For each (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We can vectorize over r
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i * m + j, :]
                    v_row = V[j * p + k, :]
                    w_row = W[k * n + i, :]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def l1_sparsify(U, V, W, max_iters=200, tol=1e-8):
        """
        L1 sparsification: iteratively minimize L1 norm of entries while preserving
        the tensor identity. Uses coordinate descent on the constraint manifold.
        """
        if not time_ok():
            return U, V, W

        rank = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()

        # Precompute for fast verification
        def check_residual():
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i * m + j, :]
                        v_row = V[j * p + k, :]
                        w_row = W[k * n + i, :]
                        val = np.dot(u_row, v_row * w_row)
                        res = abs(val - 1.0)
                        if res > max_res:
                            max_res = res
            return max_res

        # Iterative L1 sparsification using projected gradient
        # We minimize ||U||_1 + ||V||_1 + ||W||_1 subject to tensor identity
        # Use simple iterative approach: for each entry, try to shrink it toward zero
        # while maintaining feasibility

        step_size = 0.01
        for iteration in range(max_iters):
            if not time_ok():
                break

            improved = False

            # Try shrinking U entries
            for i in range(n * m):
                for r in range(rank):
                    if abs(U[i, r]) < tol:
                        continue
                    old_val = U[i, r]
                    # Try shrinking by step_size toward zero
                    shrink = min(step_size, abs(old_val) * 0.1)
                    new_val = old_val - np.sign(old_val) * shrink
                    U[i, r] = new_val

                    # Check if still feasible (approximate)
                    # Quick check: verify a few random (i,j,k) tuples
                    feasible = True
                    for _ in range(3):
                        ii = np.random.randint(n)
                        jj = np.random.randint(m)
                        kk = np.random.randint(p)
                        if ii * m + jj != i:
                            continue
                        u_row = U[ii * m + jj, :]
                        v_row = V[jj * p + kk, :]
                        w_row = W[kk * n + ii, :]
                        val = np.dot(u_row, v_row * w_row)
                        if abs(val - 1.0) > 1e-6:
                            feasible = False
                            break

                    if not feasible:
                        U[i, r] = old_val
                    else:
                        improved = True

            # Try shrinking V entries
            for j in range(m * p):
                for r in range(rank):
                    if abs(V[j, r]) < tol:
                        continue
                    old_val = V[j, r]
                    shrink = min(step_size, abs(old_val) * 0.1)
                    new_val = old_val - np.sign(old_val) * shrink
                    V[j, r] = new_val

                    feasible = True
                    for _ in range(3):
                        ii = np.random.randint(n)
                        jj = np.random.randint(m)
                        kk = np.random.randint(p)
                        if jj * p + kk != j:
                            continue
                        u_row = U[ii * m + jj, :]
                        v_row = V[jj * p + kk, :]
                        w_row = W[kk * n + ii, :]
                        val = np.dot(u_row, v_row * w_row)
                        if abs(val - 1.0) > 1e-6:
                            feasible = False
                            break

                    if not feasible:
                        V[j, r] = old_val
                    else:
                        improved = True

            # Try shrinking W entries
            for k in range(p * n):
                for r in range(rank):
                    if abs(W[k, r]) < tol:
                        continue
                    old_val = W[k, r]
                    shrink = min(step_size, abs(old_val) * 0.1)
                    new_val = old_val - np.sign(old_val) * shrink
                    W[k, r] = new_val

                    feasible = True
                    for _ in range(3):
                        ii = np.random.randint(n)
                        jj = np.random.randint(m)
                        kk = np.random.randint(p)
                        if kk * n + ii != k:
                            continue
                        u_row = U[ii * m + jj, :]
                        v_row = V[jj * p + kk, :]
                        w_row = W[kk * n + ii, :]
                        val = np.dot(u_row, v_row * w_row)
                        if abs(val - 1.0) > 1e-6:
                            feasible = False
                            break

                    if not feasible:
                        W[k, r] = old_val
                    else:
                        improved = True

            if not improved:
                break

        # Final cleanup: zero out very small entries
        U[np.abs(U) < tol] = 0.0
        V[np.abs(V) < tol] = 0.0
        W[np.abs(W) < tol] = 0.0

        return U, V, W

    def greedy_column_merge(U, V, W):
        """
        Try to merge pairs of columns that are linearly dependent or nearly so.
        If two columns r1, r2 can be combined into one column, we reduce rank by 1.
        """
        if not time_ok():
            return U, V, W

        rank = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()

        # Try to find columns that can be merged
        # Two columns r1, r2 can be merged if there exists a linear combination
        # that preserves the tensor identity.
        # Simple heuristic: if column r2 is a scalar multiple of column r1 in all three matrices,
        # we can drop r2.

        removed = set()
        for r1 in range(rank):
            if r1 in removed:
                continue
            if not time_ok():
                break
            for r2 in range(r1 + 1, rank):
                if r2 in removed:
                    continue
                if not time_ok():
                    break

                # Check if column r2 is a scalar multiple of column r1
                # Find the ratio from the first non-zero entry
                ratio = None
                for i in range(n * m):
                    if abs(U[i, r1]) > 1e-10:
                        if abs(U[i, r2]) > 1e-10:
                            ratio = U[i, r2] / U[i, r1]
                            break
                        else:
                            # U[i,r2] is 0 but U[i,r1] is not, so not a multiple
                            ratio = None
                            break

                if ratio is None:
                    continue

                # Verify the ratio holds for all entries
                is_multiple = True
                for i in range(n * m):
                    if abs(U[i, r1] * ratio - U[i, r2]) > 1e-8:
                        is_multiple = False
                        break
                if is_multiple:
                    for j in range(m * p):
                        if abs(V[j, r1] * ratio - V[j, r2]) > 1e-8:
                            is_multiple = False
                            break
                if is_multiple:
                    for k in range(p * n):
                        if abs(W[k, r1] * ratio - W[k, r2]) > 1e-8:
                            is_multiple = False
                            break

                if is_multiple:
                    removed.add(r2)

        if not removed:
            return U, V, W

        # Remove the identified columns
        keep = [r for r in range(rank) if r not in removed]
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]

        # Verify
        if verify_fast(U_new, V_new, W_new):
            return U_new, V_new, W_new
        else:
            return U, V, W

    def greedy_column_remove(U, V, W):
        """
        Try to remove individual columns if the decomposition remains valid.
        This is more aggressive than merge.
        """
        if not time_ok():
            return U, V, W

        rank = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()

        # Try removing each column
        for r in range(rank - 1, -1, -1):
            if not time_ok():
                break

            # Remove column r
            keep = [c for c in range(U.shape[1]) if c != r]
            U_test = U[:, keep]
            V_test = V[:, keep]
            W_test = W[:, keep]

            if verify_fast(U_test, V_test, W_test):
                U, V, W = U_test, V_test, W_test

        return U, V, W

    def try_column_combination(U, V, W, max_pairs=100):
        """
        Try linear combinations of column pairs to see if we can reduce rank.
        For each pair (r1, r2), try replacing them with a single column that is
        a linear combination, and check if the decomposition still holds.
        """
        if not time_ok():
            return U, V, W

        rank = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()

        # For each pair, try to find alpha, beta such that the tensor identity holds
        # with the two columns replaced by one.
        # This is complex, so we use a simpler heuristic.

        return U, V, W

    # Main logic
    best_U, best_V, best_W, best_rank = None, None, None, float('inf')

    # Try multiple structural variants
    for variant in [0, 1]:
        if not time_ok():
            break

        terms = build_structural_decomposition(variant)
        U, V, W, rank = terms_to_matrices(terms)

        # Verify
        if not verify_fast(U, V, W):
            continue

        # Apply L1 sparsification
        U_sp, V_sp, W_sp = l1_sparsify(U, V, W, max_iters=100)

        # Verify after sparsification
        if verify_fast(U_sp, V_sp, W_sp):
            # Try column merge
            U_m, V_m, W_m = greedy_column_merge(U_sp, V_sp, W_sp)
            if verify_fast(U_m, V_m, W_m) and U_m.shape[1] < best_rank:
                best_U, best_V, best_W, best_rank = U_m, V_m, W_m, U_m.shape[1]

            # Try column removal
            U_r, V_r, W_r = greedy_column_remove(U_m, V_m, W_m)
            if verify_fast(U_r, V_r, W_r) and U_r.shape[1] < best_rank:
                best_U, best_V, best_W, best_rank = U_r, V_r, W_r, U_r.shape[1]
        elif rank < best_rank:
            # Keep the original if sparsification broke it
            best_U, best_V, best_W, best_rank = U, V, W, rank

    # If no valid decomposition found, fall back to schoolbook
    if best_U is None:
        terms = build_structural_decomposition(0)
        best_U, best_V, best_W, best_rank = terms_to_matrices(terms)

    # Final cleanup
    best_U[np.abs(best_U) < 1e-10] = 0.0
    best_V[np.abs(best_V) < 1e-10] = 0.0
    best_W[np.abs(best_W) < 1e-10] = 0.0

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
