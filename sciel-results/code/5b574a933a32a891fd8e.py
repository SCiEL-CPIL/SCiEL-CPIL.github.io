import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8

    start_time = time.time()
    deadline = start_time + 130.0

    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomp(U_mat, V_mat, W_mat, rank_val):
        S = np.zeros((nm, mp, pn), dtype=float)
        for r in range(rank_val):
            S += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        return np.max(np.abs(S - T))

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2.0
                if abs(rounded - val) < 1e-10:
                    row_list.append(rounded)
                else:
                    row_list.append(val)
            result.append(row_list)
        return result

    # Build rank-28 Strassen solution
    def strassen_2x2_patterns():
        # M1 = (a11+a22)(b11+b22), c11+=1, c22+=1, c21-=1
        # M2 = (a21+a22)(b11), c21+=1, c22-=1
        # M3 = (a11)(b12-b22), c12+=1, c22+=1
        # M4 = (a22)(b21-b11), c11+=1, c21+=1
        # M5 = (a11+a12)(b22), c11-=1, c12+=1
        # M6 = (a21-a11)(b11+b12), c22+=1
        # M7 = (a12-a22)(b21+b22), c11+=1
        patterns = [
            {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1, 2: -1}},
            {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
            {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
            {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
            {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
            {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
            {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
        ]
        return patterns

    patterns = strassen_2x2_patterns()
    rank = 28
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    # Four blocks: (a_base_j, b_base_j, b_base_k, c_base_k)
    blocks = [
        (0, 0, 0, 0),  # A1*B11 -> C[:, 0:2]
        (2, 2, 0, 0),  # A2*B21 -> C[:, 0:2]
        (0, 0, 2, 2),  # A1*B12 -> C[:, 2:4]
        (2, 2, 2, 2),  # A2*B22 -> C[:, 2:4]
    ]

    col = 0
    for (a_bj, b_bj, b_bk, c_bk) in blocks:
        for pattern in patterns:
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = local_r
                global_j = a_bj + local_c
                U[global_i * m + global_j, col] = coeff

            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_bj + local_r
                global_k = b_bk + local_c
                V[global_j * p + global_k, col] = coeff

            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = local_r
                global_k = c_bk + local_c
                # Trace convention: W[k*n + i]
                W[global_k * n + global_i, col] = coeff
            col += 1

    # Verify rank-28
    max_err = verify_decomp(U, V, W, rank)
    if max_err > 1e-10:
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()

    def build_term_matrix(U_mat, V_mat, W_mat, r):
        M = np.zeros((nm * mp * pn, r), dtype=float)
        for ri in range(r):
            M[:, ri] = np.einsum('a,b,c->abc', U_mat[:, ri], V_mat[:, ri], W_mat[:, ri]).flatten()
        return M

    def try_remove_columns(U_cur, V_cur, W_cur, cols_to_remove):
        remaining = [c for c in range(U_cur.shape[1]) if c not in cols_to_remove]
        if len(remaining) == 0:
            return None
        U_new = U_cur[:, remaining]
        V_new = V_cur[:, remaining]
        W_new = W_cur[:, remaining]
        err = verify_decomp(U_new, V_new, W_new, len(remaining))
        if err < 1e-10:
            return (U_new, V_new, W_new)
        return None

    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank

    # Aggressive rank reduction: check for linear dependencies
    for iteration in range(30):
        if time.time() > deadline - 20:
            break
        if current_rank <= 27:
            break

        M = build_term_matrix(current_U, current_V, current_W, current_rank)
        
        # Check SVD for rank deficiency
        try:
            s = np.linalg.svd(M, compute_uv=False)
            # If smallest singular value is very small, there's a dependency
            if s[-1] < 1e-8 * s[0]:
                # Find which column to remove - the one most aligned with null space
                # Use least squares to find dependency
                for r_elim in range(current_rank):
                    if time.time() > deadline - 15:
                        break
                    other_cols = [s_idx for s_idx in range(current_rank) if s_idx != r_elim]
                    M_sub = M[:, other_cols]
                    rhs = M[:, r_elim]
                    alpha, residuals, rk, sv = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
                    residual = np.max(np.abs(M_sub @ alpha - rhs))
                    if residual < 1e-8:
                        result = try_remove_columns(current_U, current_V, current_W, [r_elim])
                        if result is not None:
                            current_U, current_V, current_W = result
                            current_rank -= 1
                            if current_rank < best_rank:
                                best_rank = current_rank
                                best_U = current_U.copy()
                                best_V = current_V.copy()
                                best_W = current_W.copy()
                            break
                continue
        except np.linalg.LinAlgError:
            pass

        # Try removing each column
        removed_any = False
        for r_elim in range(current_rank):
            if time.time() > deadline - 15:
                break
            other_cols = [s_idx for s_idx in range(current_rank) if s_idx != r_elim]
            M_sub = M[:, other_cols]
            rhs = M[:, r_elim]
            try:
                alpha, residuals, rk, sv = np.linalg.lstsq(M_sub, rhs, rcond=1e-10)
                residual = np.max(np.abs(M_sub @ alpha - rhs))
                if residual < 1e-8:
                    result = try_remove_columns(current_U, current_V, current_W, [r_elim])
                    if result is not None:
                        current_U, current_V, current_W = result
                        current_rank -= 1
                        removed_any = True
                        if current_rank < best_rank:
                            best_rank = current_rank
                            best_U = current_U.copy()
                            best_V = current_V.copy()
                            best_W = current_W.copy()
                        break
            except np.linalg.LinAlgError:
                continue

        if not removed_any:
            break

    # Try rank-27 via MPS-inspired approach:
    # The idea is that the 2x4x4 multiplication tensor might have rank 27
    # We try to find a rank-27 decomposition by solving for the missing term
    
    if best_rank == 28 and time.time() < deadline - 30:
        # Try to find rank-27 by checking if any subset of 27 terms can span the tensor
        # This is expensive, so we use a heuristic: try removing each term and see
        # if the residual can be expressed as a linear combination of the remaining
        
        M_full = build_term_matrix(best_U, best_V, best_W, 28)
        
        # For each possible term to remove, check if the tensor can be represented
        # with the remaining 27 terms (allowing coefficient changes)
        for r_skip in range(28):
            if time.time() > deadline - 25:
                break
            remaining = [c for c in range(28) if c != r_skip]
            M_sub = M_full[:, remaining]
            
            # Check if T is in the column space of M_sub
            # Project T onto the column space
            try:
                # Use SVD to project
                U_svd, s_vals, Vt = np.linalg.svd(M_sub, full_matrices=False)
                # Project T onto the span
                T_flat = T.flatten()
                # Coefficients in the SVD basis
                coeffs = Vt[:27, :] @ T_flat  # This isn't quite right
                # Actually, we need to solve M_sub @ x = T_flat
                x, residuals, rk, sv = np.linalg.lstsq(M_sub, T_flat, rcond=1e-10)
                residual = np.max(np.abs(M_sub @ x - T_flat))
                
                if residual < 1e-8:
                    # We can represent T with 27 terms!
                    # But we need to construct the actual U, V, W matrices
                    # This is hard because the decomposition isn't unique
                    # For now, just note that rank 27 is achievable
                    pass
            except np.linalg.LinAlgError:
                continue

    # Final verification
    final_err = verify_decomp(best_U, best_V, best_W, best_rank)
    if final_err > 1e-10:
        best_U = U
        best_V = V
        best_W = W
        best_rank = 28

    U_final = to_list(best_U)
    V_final = to_list(best_V)
    W_final = to_list(best_W)

    return {"U": U_final, "V": V_final, "W": W_final}
