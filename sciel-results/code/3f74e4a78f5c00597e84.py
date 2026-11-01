import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 140  # 10s safety margin before 150s kill
    
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
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
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
    
    # Build the rank-28 solution
    U, V, W = build_rank28()
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt to reduce rank by checking for linear dependencies among the 28 terms
    # Build the 4096x28 matrix M where M[:,s] = vec(U_s ⊗ V_s ⊗ W_s)
    n_terms = 28
    dim = n * m * m * p * p * n  # 32 * 16 * 8 = 4096
    
    M = np.zeros((dim, n_terms))
    for s in range(n_terms):
        # vec(U_s ⊗ V_s ⊗ W_s) indexed by (i, j, k) -> i*(m*p)*(p*n) + j*(p*n) + k
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    M[i * (m * p) * (p * n) + j * (p * n) + k, s] = U[i, s] * V[j, s] * W[k, s]
    
    # Check numerical rank
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-9)
        if rank_M < 28 and time.time() < deadline:
            # Find null space to try to eliminate a term
            _, s_vals, vh = np.linalg.svd(M, full_matrices=False)
            null_dim = 28 - rank_M
            for nv_idx in range(null_dim):
                null_vec = vh[rank_M + nv_idx, :]
                max_idx = np.argmax(np.abs(null_vec))
                if abs(null_vec[max_idx]) < 1e-10:
                    continue
                c_new = np.ones(n_terms) + null_vec
                if abs(c_new[max_idx]) < 1e-10:
                    # Try removing term max_idx
                    U_new = np.zeros((n * m, 27), dtype=float)
                    V_new = np.zeros((m * p, 27), dtype=float)
                    W_new = np.zeros((p * n, 27), dtype=float)
                    col_map = [s for s in range(n_terms) if s != max_idx]
                    for new_col, old_col in enumerate(col_map):
                        U_new[:, new_col] = U[:, old_col] * c_new[old_col]
                        V_new[:, new_col] = V[:, old_col] * c_new[old_col]
                        W_new[:, new_col] = W[:, old_col] * c_new[old_col]
                    
                    # Verify with test matrices
                    A_test = np.arange(n * m).reshape(n, m) + 1.0
                    B_test = np.arange(m * p).reshape(m, p) + 2.0
                    C_expected = A_test @ B_test
                    
                    C_actual = np.zeros((n, p))
                    for ii in range(n):
                        for kk in range(p):
                            acc = 0.0
                            for cc in range(27):
                                for jj in range(m):
                                    acc += U_new[ii * m + jj, cc] * V_new[jj * p + kk, cc] * W_new[kk * n + ii, cc]
                            C_actual[ii, kk] = acc
                    
                    if np.allclose(C_expected, C_actual, atol=1e-9):
                        # Round to half-integers if possible
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
