import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 140.0

    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Known rank-28 solution via Strassen on 2x2 blocks
    rank = 28
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U[X_rows[0], col_offset + 0] += 1
        U[X_rows[3], col_offset + 0] += 1
        V[Y_rows[0], col_offset + 0] += 1
        V[Y_rows[3], col_offset + 0] += 1
        W[Z_rows[0], col_offset + 0] += 1
        W[Z_rows[3], col_offset + 0] += 1

        U[X_rows[2], col_offset + 1] += 1
        U[X_rows[3], col_offset + 1] += 1
        V[Y_rows[0], col_offset + 1] += 1
        W[Z_rows[2], col_offset + 1] += 1
        W[Z_rows[3], col_offset + 1] -= 1

        U[X_rows[0], col_offset + 2] += 1
        V[Y_rows[1], col_offset + 2] += 1
        V[Y_rows[3], col_offset + 2] -= 1
        W[Z_rows[1], col_offset + 2] += 1
        W[Z_rows[3], col_offset + 2] += 1

        U[X_rows[3], col_offset + 3] += 1
        V[Y_rows[2], col_offset + 3] += 1
        V[Y_rows[0], col_offset + 3] -= 1
        W[Z_rows[0], col_offset + 3] += 1
        W[Z_rows[2], col_offset + 3] += 1

        U[X_rows[0], col_offset + 4] += 1
        U[X_rows[1], col_offset + 4] += 1
        V[Y_rows[3], col_offset + 4] += 1
        W[Z_rows[0], col_offset + 4] -= 1
        W[Z_rows[1], col_offset + 4] += 1

        U[X_rows[2], col_offset + 5] += 1
        U[X_rows[0], col_offset + 5] -= 1
        V[Y_rows[0], col_offset + 5] += 1
        V[Y_rows[1], col_offset + 5] += 1
        W[Z_rows[3], col_offset + 5] += 1

        U[X_rows[1], col_offset + 6] += 1
        U[X_rows[3], col_offset + 6] -= 1
        V[Y_rows[2], col_offset + 6] += 1
        V[Y_rows[3], col_offset + 6] += 1
        W[Z_rows[0], col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    def verify_decomp(U_mat, V_mat, W_mat, rank_val):
        S = np.zeros((nm, mp, pn), dtype=float)
        for r in range(rank_val):
            S += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        return np.max(np.abs(S - T))

    max_err = verify_decomp(U, V, W, rank)
    if max_err > 1e-10:
        # Fallback: schoolbook
        rank = 32
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    r = i * m * p + l * p + k
                    a = i * m + l
                    b = l * p + k
                    c = k * n + i
                    U[a, r] = 1.0
                    V[b, r] = 1.0
                    W[c, r] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find linear dependencies and eliminate them correctly
    # M is a (nm*mp*pn) x rank matrix where each column is the flattened rank-1 tensor
    M = np.zeros((nm * mp * pn, rank), dtype=float)
    for r in range(rank):
        M[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()

    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()

    # Iteratively eliminate linearly dependent terms
    # If sum(alpha_i * v_i) = 0 where v_i is the i-th rank-1 tensor,
    # we can eliminate term j by scaling all other terms: new_coeff_i = coeff_i + alpha_i/alpha_j * coeff_j
    # But wait - the terms are the U,V,W columns themselves, not coefficients of a fixed basis.
    # The decomposition is T = sum_r U[:,r] * V[:,r] * W[:,r] (outer products)
    # If there's a linear dependency sum(alpha_r * T_r) = 0 where T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r],
    # then we can rewrite: T = sum_r T_r = sum_{r != j} T_r + T_j = sum_{r != j} (T_r + alpha_r/alpha_j * T_j)
    # But T_r + alpha_r/alpha_j * T_j is NOT a rank-1 tensor in general!
    # 
    # Actually, the correct approach: if sum(alpha_r * T_r) = 0, then
    # T = sum_r T_r. Pick j with alpha_j != 0. Then T_j = -sum_{r != j} alpha_r/alpha_j * T_r.
    # So T = sum_{r != j} T_r - sum_{r != j} alpha_r/alpha_j * T_r = sum_{r != j} (1 - alpha_r/alpha_j) * T_r.
    # This is still a sum of rank-1 tensors (with scaled coefficients), so rank decreases by 1.
    # We just multiply column r of U by (1 - alpha_r/alpha_j) for r != j, and drop column j.

    current_rank = rank
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_M = M.copy()

    for elimination_round in range(rank):
        if time.time() > deadline - 20:
            break
        if current_rank <= 1:
            break

        # Check rank of M
        try:
            sv_vals = np.linalg.svd(current_M, compute_uv=False)
            tol = 1e-10
            effective_rank = np.sum(sv_vals > tol)
        except:
            break

        if effective_rank >= current_rank:
            break

        # Find null space
        try:
            U_svd, s, Vt_svd = np.linalg.svd(current_M, full_matrices=True)
            # Null space vectors are the last (current_rank - effective_rank) rows of Vt_svd
            null_dim = current_rank - effective_rank
            null_space = Vt_svd[effective_rank:, :].T  # shape (current_rank, null_dim)
        except:
            break

        # Try each null space vector to find one we can use for elimination
        eliminated_this_round = False
        for ns_vec in null_space.T:
            if time.time() > deadline - 15:
                break
            if np.max(np.abs(ns_vec)) < 1e-10:
                continue
            # Find index with largest magnitude
            j = np.argmax(np.abs(ns_vec))
            if abs(ns_vec[j]) < 1e-10:
                continue
            alpha_j = ns_vec[j]

            # Verify the dependency: sum(ns_vec[r] * T_r) ≈ 0
            dep_check = np.zeros((nm, mp, pn), dtype=float)
            for r in range(current_rank):
                if abs(ns_vec[r]) > 1e-15:
                    dep_check += ns_vec[r] * np.einsum('a,b,c->abc', current_U[:, r], current_V[:, r], current_W[:, r])
            if np.max(np.abs(dep_check)) > 1e-8:
                continue

            # Eliminate term j: scale all other terms by (1 - ns_vec[r]/ns_vec[j])
            new_U = np.zeros((nm, current_rank - 1), dtype=float)
            new_V = np.zeros((mp, current_rank - 1), dtype=float)
            new_W = np.zeros((pn, current_rank - 1), dtype=float)

            col_idx = 0
            for r in range(current_rank):
                if r == j:
                    continue
                scale = 1.0 - ns_vec[r] / alpha_j
                new_U[:, col_idx] = current_U[:, r] * scale
                new_V[:, col_idx] = current_V[:, r] * scale
                new_W[:, col_idx] = current_W[:, r] * scale
                col_idx += 1

            # Verify
            err = verify_decomp(new_U, new_V, new_W, current_rank - 1)
            if err < 1e-10:
                current_rank -= 1
                current_U = new_U
                current_V = new_V
                current_W = new_W
                # Recompute M
                current_M = np.zeros((nm * mp * pn, current_rank), dtype=float)
                for r in range(current_rank):
                    current_M[:, r] = np.einsum('a,b,c->abc', current_U[:, r], current_V[:, r], current_W[:, r]).flatten()
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                eliminated_this_round = True
                break

        if not eliminated_this_round:
            break

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Round to nearest half-integer if close
                rounded = round(val * 2) / 2.0
                if abs(rounded - val) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
