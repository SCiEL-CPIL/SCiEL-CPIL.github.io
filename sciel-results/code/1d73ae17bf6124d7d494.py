import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    
    # Build rank-28 Strassen decomposition
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
    
    # Check for linear dependencies to reduce rank
    # Flatten U, V, W columns into vectors
    # T_r = U[:,r] \otimes V[:,r] \otimes W[:,r]
    # We want to see if any T_r is in the span of the others.
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total_dim = dim_u * dim_v * dim_w
    
    T_flat = np.zeros((total_dim, rank))
    for r in range(rank):
        T_flat[:, r] = np.kron(np.kron(U[:, r], V[:, r]), W[:, r])
        
    # Check rank of T_flat
    # If rank < 28, there are dependencies.
    # We want to find a subset of columns that still spans the same tensor T?
    # No, T is the sum of the columns.
    # If T_r is in span of others, we can eliminate it.
    
    # Let's try to find a minimal subset.
    # This is a combinatorial problem.
    # Heuristic: Try to remove one column at a time if it is in the span of the remaining.
    
    active = list(range(rank))
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(active)):
            # Try removing active[i]
            new_active = active[:i] + active[i+1:]
            if not new_active:
                continue
            # Check if T[active[i]] is in span of T[new_active]
            # Solve T[new_active] x = T[active[i]]
            try:
                # Use least squares to check if it's in the span
                A_mat = T_flat[:, new_active]
                b_vec = T_flat[:, active[i]]
                # If rank(A_mat) < len(new_active), use pinv
                x, residuals, rank_mat, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                # Check if residual is small
                if np.linalg.norm(A_mat @ x - b_vec) < 1e-9:
                    # It is in the span. Remove it.
                    active = new_active
                    changed = True
                    break
            except np.linalg.LinAlgError:
                pass
                
    if active:
        U_red = U[:, active]
        V_red = V[:, active]
        W_red = W[:, active]
        # Verify
        def check_decomposition(U_mat, V_mat, W_mat):
            rng = np.random.RandomState(42)
            A_test = rng.randn(n, m)
            B_test = rng.randn(m, p)
            C_expected = A_test @ B_test
            C_computed = np.zeros((n, p))
            for ii in range(n):
                for kk in range(p):
                    acc = 0.0
                    for cc in range(U_mat.shape[1]):
                        for jj in range(m):
                            acc += U_mat[ii * m + jj, cc] * V_mat[jj * p + kk, cc] * W_mat[kk * n + ii, cc]
                    C_computed[ii, kk] = acc
            return np.allclose(C_expected, C_computed, atol=1e-10)
            
        if check_decomposition(U_red, V_red, W_red):
            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
