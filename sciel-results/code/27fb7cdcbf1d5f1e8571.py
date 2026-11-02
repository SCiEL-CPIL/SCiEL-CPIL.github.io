import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 130  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    def strassen_terms():
        terms = []
        terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
        terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}))
        terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
        terms.append(({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
        terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
        terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
        terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}))
        return terms

    base_terms = strassen_terms()

    def build_rank28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        def add_term(u_vec, v_vec, w_vec, col_idx):
            for idx, coeff in u_vec:
                U[idx, col_idx] += coeff
            for idx, coeff in v_vec:
                V[idx, col_idx] += coeff
            for idx, coeff in w_vec:
                W[idx, col_idx] += coeff

        def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
            for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
                col = start_col + t_idx

                u_global = {}
                for loc_idx, coeff in u_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = r
                    g_c = a_offset * 2 + c
                    g_idx = g_r * 4 + g_c
                    u_global[g_idx] = u_global.get(g_idx, 0) + coeff

                v_global = {}
                for loc_idx, coeff in v_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = b_row_offset * 2 + r
                    g_c = b_col_offset * 2 + c
                    g_idx = g_r * 4 + g_c
                    v_global[g_idx] = v_global.get(g_idx, 0) + coeff

                w_global = {}
                for loc_idx, coeff in w_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = r
                    g_c = c_col_offset * 2 + c
                    g_idx = g_c * 2 + g_r
                    w_global[g_idx] = w_global.get(g_idx, 0) + coeff

                add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)

        apply_strassen(0, 0, 0, 0, 0)
        apply_strassen(1, 1, 0, 0, 7)
        apply_strassen(0, 0, 1, 1, 14)
        apply_strassen(1, 1, 1, 1, 21)

        return U, V, W

    def verify(U, V, W, A_test, B_test):
        C_expected = A_test @ B_test
        rank = U.shape[1]
        C_actual = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(rank):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                C_actual[ii, kk] = acc
        return np.allclose(C_expected, C_actual, atol=1e-9)

    # Build the rank-28 solution
    U, V, W = build_rank28()

    A_test = np.arange(n * m).reshape(n, m).astype(float) + 1.0
    B_test = np.arange(m * p).reshape(m, p).astype(float) + 2.0

    if time.time() > deadline - 30:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the tensor matrix M where M[:,s] = vec(U_s ⊗ V_s ⊗ W_s)
    # More efficient: use the fact that M[i,j,k,s] = U[i,s]*V[j,s]*W[k,s]
    # We'll build M as a 3D array reshaped to 2D
    n_terms = 28
    dim = n * m * m * p * p * n  # 32 * 16 * 8 = 4096

    # Efficient construction using broadcasting
    # M_flat has shape (dim, n_terms)
    # M_flat[i*(m*p)*(p*n) + j*(p*n) + k, s] = U[i,s] * V[j,s] * W[k,s]
    
    # Build using outer products
    M = np.zeros((dim, n_terms), dtype=float)
    
    for s in range(n_terms):
        u_s = U[:, s]  # (n*m,)
        v_s = V[:, s]  # (m*p,)
        w_s = W[:, s]  # (p*n,)
        
        # For each i, j: M[i*(m*p)*(p*n) + j*(p*n) + (0..p*n-1), s] = u_s[i]*v_s[j]*w_s
        for i in range(n * m):
            for j in range(m * p):
                base_idx = i * (m * p) * (p * n) + j * (p * n)
                M[base_idx:base_idx + p * n, s] = u_s[i] * v_s[j] * w_s

    if time.time() > deadline - 25:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Check numerical rank and try to reduce
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-10)
        
        if rank_M < 28 and time.time() < deadline - 15:
            # Find null space
            _, s_vals, vh = np.linalg.svd(M, full_matrices=False)
            null_dim = 28 - rank_M

            best_result = None

            for nv_idx in range(null_dim):
                if time.time() > deadline - 10:
                    break

                null_vec = vh[rank_M + nv_idx, :]
                max_idx = np.argmax(np.abs(null_vec))
                if abs(null_vec[max_idx]) < 1e-10:
                    continue

                # Try removing each term that has significant coefficient in null vector
                for remove_idx in range(n_terms):
                    if abs(null_vec[remove_idx]) < 1e-8:
                        continue

                    c_new = np.ones(n_terms) + null_vec
                    if abs(c_new[remove_idx]) > 1e-6:
                        continue

                    # Build new factor matrices with 27 terms
                    U_new = np.zeros((n * m, 27), dtype=float)
                    V_new = np.zeros((m * p, 27), dtype=float)
                    W_new = np.zeros((p * n, 27), dtype=float)
                    col_map = [s for s in range(n_terms) if s != remove_idx]
                    for new_col, old_col in enumerate(col_map):
                        U_new[:, new_col] = U[:, old_col] * c_new[old_col]
                        V_new[:, new_col] = V[:, old_col] * c_new[old_col]
                        W_new[:, new_col] = W[:, old_col] * c_new[old_col]

                    if verify(U_new, V_new, W_new, A_test, B_test):
                        U_out = np.round(U_new * 2) / 2
                        V_out = np.round(V_new * 2) / 2
                        W_out = np.round(W_new * 2) / 2

                        if (np.max(np.abs(U_new - U_out)) < 1e-9 and
                            np.max(np.abs(V_new - V_out)) < 1e-9 and
                            np.max(np.abs(W_new - W_out)) < 1e-9):
                            return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
                        else:
                            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

    except Exception:
        pass

    # Alternative approach: try all single-term removals via least squares
    if time.time() < deadline - 10:
        try:
            for remove_idx in range(n_terms):
                if time.time() > deadline - 5:
                    break

                M_without = np.delete(M, remove_idx, axis=1)
                col_to_remove = M[:, remove_idx]

                try:
                    x, residuals, rank_r, svals = np.linalg.lstsq(M_without, col_to_remove, rcond=1e-12)
                    if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-10:
                        c_new = np.ones(n_terms)
                        for new_idx, old_idx in enumerate([s for s in range(n_terms) if s != remove_idx]):
                            c_new[old_idx] = 1.0 + x[new_idx]
                        c_new[remove_idx] = 0.0

                        U_new = np.zeros((n * m, 27), dtype=float)
                        V_new = np.zeros((m * p, 27), dtype=float)
                        W_new = np.zeros((p * n, 27), dtype=float)
                        col_map = [s for s in range(n_terms) if s != remove_idx]
                        for new_col, old_col in enumerate(col_map):
                            U_new[:, new_col] = U[:, old_col] * c_new[old_col]
                            V_new[:, new_col] = V[:, old_col] * c_new[old_col]
                            W_new[:, new_col] = W[:, old_col] * c_new[old_col]

                        if verify(U_new, V_new, W_new, A_test, B_test):
                            U_out = np.round(U_new * 2) / 2
                            V_out = np.round(V_new * 2) / 2
                            W_out = np.round(W_new * 2) / 2

                            if (np.max(np.abs(U_new - U_out)) < 1e-9 and
                                np.max(np.abs(V_new - V_out)) < 1e-9 and
                                np.max(np.abs(W_new - W_out)) < 1e-9):
                                return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
                            else:
                                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                except Exception:
                    continue
        except Exception:
            pass

    # Try structured perturbation: slightly modify Strassen coefficients and re-check rank
    if time.time() < deadline - 8:
        try:
            # Try small integer perturbations to see if we can find a rank-27 solution
            # by adjusting the relative weights of the 4 Strassen blocks
            for pert in [0.5, -0.5, 0.25, -0.25]:
                if time.time() > deadline - 5:
                    break
                    
                U_p = U.copy()
                V_p = V.copy()
                W_p = W.copy()
                
                # Scale the second block (cols 7-13) by (1+pert)
                for c in range(7, 14):
                    U_p[:, c] *= (1 + pert)
                    V_p[:, c] *= (1 + pert)
                    W_p[:, c] *= (1 + pert)
                
                # Rebuild M and check rank
                M_p = np.zeros((dim, n_terms), dtype=float)
                for s in range(n_terms):
                    u_s = U_p[:, s]
                    v_s = V_p[:, s]
                    w_s = W_p[:, s]
                    for i in range(n * m):
                        for j in range(m * p):
                            base_idx = i * (m * p) * (p * n) + j * (p * n)
                            M_p[base_idx:base_idx + p * n, s] = u_s[i] * v_s[j] * w_s
                
                rank_p = np.linalg.matrix_rank(M_p, tol=1e-10)
                if rank_p < 28:
                    _, s_vals_p, vh_p = np.linalg.svd(M_p, full_matrices=False)
                    null_dim_p = 28 - rank_p
                    for nv_idx in range(null_dim_p):
                        null_vec_p = vh_p[rank_p + nv_idx, :]
                        for remove_idx in range(n_terms):
                            if abs(null_vec_p[remove_idx]) < 1e-8:
                                continue
                            c_new_p = np.ones(n_terms) + null_vec_p
                            if abs(c_new_p[remove_idx]) > 1e-6:
                                continue
                            
                            U_new_p = np.zeros((n * m, 27), dtype=float)
                            V_new_p = np.zeros((m * p, 27), dtype=float)
                            W_new_p = np.zeros((p * n, 27), dtype=float)
                            col_map_p = [s for s in range(n_terms) if s != remove_idx]
                            for new_col, old_col in enumerate(col_map_p):
                                U_new_p[:, new_col] = U_p[:, old_col] * c_new_p[old_col]
                                V_new_p[:, new_col] = V_p[:, old_col] * c_new_p[old_col]
                                W_new_p[:, new_col] = W_p[:, old_col] * c_new_p[old_col]
                            
                            if verify(U_new_p, V_new_p, W_new_p, A_test, B_test):
                                U_out_p = np.round(U_new_p * 2) / 2
                                V_out_p = np.round(V_new_p * 2) / 2
                                W_out_p = np.round(W_new_p * 2) / 2
                                
                                if (np.max(np.abs(U_new_p - U_out_p)) < 1e-9 and
                                    np.max(np.abs(V_new_p - V_out_p)) < 1e-9 and
                                    np.max(np.abs(W_new_p - W_out_p)) < 1e-9):
                                    return {"U": U_out_p.tolist(), "V": V_out_p.tolist(), "W": W_out_p.tolist()}
                                else:
                                    return {"U": U_new_p.tolist(), "V": V_new_p.tolist(), "W": W_new_p.tolist()}
        except Exception:
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
