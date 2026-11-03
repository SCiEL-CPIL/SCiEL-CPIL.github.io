import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Strassen-based decomposition for 2x4x5, rank 36
    total_rank_36 = 36
    U36 = np.zeros((nm, total_rank_36), dtype=np.float64)
    V36 = np.zeros((mp, total_rank_36), dtype=np.float64)
    W36 = np.zeros((pn, total_rank_36), dtype=np.float64)

    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        strassen_patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    col = 0
    col = add_2x2x2_block(U36, V36, W36, 0, 0, 0, 0, 0, 0, col)
    col = add_2x2x2_block(U36, V36, W36, 0, 0, 0, 2, 0, 2, col)
    col = add_2x2x1_block(U36, V36, W36, 0, 0, 0, 4, 0, 4, col)
    col = add_2x2x2_block(U36, V36, W36, 0, 2, 2, 0, 0, 0, col)
    col = add_2x2x2_block(U36, V36, W36, 0, 2, 2, 2, 0, 2, col)
    col = add_2x2x1_block(U36, V36, W36, 0, 2, 2, 4, 0, 4, col)
    assert col == total_rank_36

    def verify_decomposition(U_mat, V_mat, W_mat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_mat[i * m + j] * V_mat[j * p + k] * W_mat[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m; j_a = a % m
                    j_b = b // p; k_b = b % p
                    k_c = c // n; i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    # Verify rank 36 solution
    if verify_decomposition(U36, V36, W36):
        U_list = [[int(round(U36[i, r] * 2)) / 2 for r in range(total_rank_36)] for i in range(nm)]
        V_list = [[int(round(V36[i, r] * 2)) / 2 for r in range(total_rank_36)] for i in range(mp)]
        W_list = [[int(round(W36[i, r] * 2)) / 2 for r in range(total_rank_36)] for i in range(pn)]
        best_result = {"U": U_list, "V": V_list, "W": W_list}
        best_rank = 36

    # Try to find a rank 35 solution directly for 2x4x5
    if time.time() < deadline - 30.0:
        try:
            from scipy.optimize import least_squares
            
            target_rank_245 = 35
            n_vars_245 = (nm + mp + pn) * target_rank_245
            
            ones_245 = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ones_245.append((i*m+j, j*p+k, k*n+i))
            
            zeros_245 = []
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        i_a = a // m; j_a = a % m
                    j_b = b // p; k_b = b % p
                    k_c = c // n; i_c = c % n
                    if not (j_a == j_b and k_b == k_c and i_a == i_c):
                        zeros_245.append((a, b, c))
            
            np.random.seed(123)
            # Use a subset of zeros to keep problem tractable
            n_zero_constraints = min(500, len(zeros_245))
            zero_subset_245 = zeros_245[:n_zero_constraints]
            
            def residuals_245(x):
                U_r = x[:nm*target_rank_245]
                V_r = x[nm*target_rank_245:nm*target_rank_245+mp*target_rank_245]
                W_r = x[nm*target_rank_245+mp*target_rank_245:n_vars_245]
                U_mat = U_r.reshape(nm, target_rank_245)
                V_mat = V_r.reshape(mp, target_rank_245)
                W_mat = W_r.reshape(pn, target_rank_245)
                
                res = []
                for a, b, c in ones_245:
                    val = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                    res.append(val - 1.0)
                for a, b, c in zero_subset_245:
                    val = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                    res.append(val)
                return res
            
            best_res_norm_245 = np.inf
            best_x_245 = None
            
            n_restarts_245 = 15
            for attempt in range(n_restarts_245):
                if time.time() > deadline - 20.0:
                    break
                
                if attempt == 0:
                    # Initialize with the rank-36 solution padded with zeros
                    init_x = np.zeros(n_vars_245)
                    for r in range(36):
                        if r < target_rank_245:
                            init_x[r*nm:(r+1)*nm] = U36[:, r]
                            init_x[nm*target_rank_245 + r*mp:nm*target_rank_245 + (r+1)*mp] = V36[:, r]
                            init_x[nm*target_rank_245 + mp*target_rank_245 + r*pn:nm*target_rank_245 + mp*target_rank_245 + (r+1)*pn] = W36[:, r]
                elif attempt == 1:
                    init_x = np.random.uniform(-0.5, 0.5, size=n_vars_245)
                elif best_x_245 is not None and attempt < 10:
                    init_x = best_x_245 + np.random.normal(0, 0.3, size=n_vars_245)
                else:
                    init_x = np.random.uniform(-1, 1, size=n_vars_245)
                
                try:
                    sol = least_squares(residuals_245, init_x, method='trf', 
                                       max_nfev=500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                    res_norm = np.linalg.norm(sol.fun)
                    if res_norm < best_res_norm_245:
                        best_res_norm_245 = res_norm
                        best_x_245 = sol.x
                        if res_norm < 1e-10:
                            break
                except Exception:
                    pass
            
            if best_x_245 is not None and best_res_norm_245 < 1e-8:
                U_245 = best_x_245[:nm*target_rank_245].reshape(nm, target_rank_245)
                V_245 = best_x_245[nm*target_rank_245:nm*target_rank_245+mp*target_rank_245].reshape(mp, target_rank_245)
                W_245 = best_x_245[nm*target_rank_245+mp*target_rank_245:n_vars_245].reshape(pn, target_rank_245)
                
                # Snap to half-integers
                U_245_s = np.round(U_245 * 2) / 2
                V_245_s = np.round(V_245 * 2) / 2
                W_245_s = np.round(W_245 * 2) / 2
                
                if verify_decomposition(U_245_s, V_245_s, W_245_s):
                    U_list = [[int(round(U_245_s[i, r] * 2)) / 2 for r in range(target_rank_245)] for i in range(nm)]
                    V_list = [[int(round(V_245_s[i, r] * 2)) / 2 for r in range(target_rank_245)] for i in range(mp)]
                    W_list = [[int(round(W_245_s[i, r] * 2)) / 2 for r in range(target_rank_245)] for i in range(pn)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = 35
                    return best_result
                    
        except Exception:
            pass

    # Try to find a rank 34 solution by finding rank 17 decompositions for 2x2x5 blocks
    if time.time() < deadline - 40.0:
        try:
            from scipy.optimize import least_squares
            
            # Find rank 17 decomposition for 2x2x5
            n2, m2, p2 = 2, 2, 5
            nm2, mp2, pn2 = n2*m2, m2*p2, p2*n2  # 4, 10, 10
            target_rank_225 = 17
            
            ones_225 = []
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        ones_225.append((i*m2+j, j*p2+k, k*n2+i))
            
            zeros_225 = []
            for a in range(nm2):
                for b in range(mp2):
                    for c in range(pn2):
                        i_a = a // m2; j_a = a % m2
                        j_b = b // p2; k_b = b % p2
                        k_c = c // n2; i_c = c % n2
                        if not (j_a == j_b and k_b == k_c and i_a == i_c):
                            zeros_225.append((a, b, c))
            
            np.random.seed(42)
            zero_subset_225 = zeros_225[:300] if len(zeros_225) > 300 else zeros_225
            
            n_vars_225 = (nm2 + mp2 + pn2) * target_rank_225
            
            def residuals_225(x):
                U_r = x[:nm2*target_rank_225]
                V_r = x[nm2*target_rank_225:nm2*target_rank_225+mp2*target_rank_225]
                W_r = x[nm2*target_rank_225+mp2*target_rank_225:n_vars_225]
                U_mat = U_r.reshape(nm2, target_rank_225)
                V_mat = V_r.reshape(mp2, target_rank_225)
                W_mat = W_r.reshape(pn2, target_rank_225)
                
                res = []
                for a, b, c in ones_225:
                    val = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                    res.append(val - 1.0)
                for a, b, c in zero_subset_225:
                    val = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                    res.append(val)
                return res
            
            best_res_norm_225 = np.inf
            best_x_225 = None
            
            n_restarts_225 = 20
            for attempt in range(n_restarts_225):
                if time.time() > deadline - 25.0:
                    break
                
                if attempt == 0:
                    init_x = np.random.uniform(-1, 1, size=n_vars_225)
                elif attempt == 1:
                    init_x = np.zeros(n_vars_225)
                    nz_idx = np.random.choice(n_vars_225, n_vars_225//4, replace=False)
                    init_x[nz_idx] = np.random.choice([-1, 0.5, 1, -0.5], size=len(nz_idx))
                elif best_x_225 is not None and attempt < 15:
                    init_x = best_x_225 + np.random.normal(0, 0.5, size=n_vars_225)
                else:
                    init_x = np.random.uniform(-2, 2, size=n_vars_225)
                
                try:
                    sol = least_squares(residuals_225, init_x, method='trf', 
                                       max_nfev=800, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                    res_norm = np.linalg.norm(sol.fun)
                    if res_norm < best_res_norm_225:
                        best_res_norm_225 = res_norm
                        best_x_225 = sol.x
                        if res_norm < 1e-10:
                            break
                except Exception:
                    pass
            
            if best_x_225 is not None and best_res_norm_225 < 1e-8:
                U_225 = best_x_225[:nm2*target_rank_225].reshape(nm2, target_rank_225)
                V_225 = best_x_225[nm2*target_rank_225:nm2*target_rank_225+mp2*target_rank_225].reshape(mp2, target_rank_225)
                W_225 = best_x_225[nm2*target_rank_225+mp2*target_rank_225:n_vars_225].reshape(pn2, target_rank_225)
                
                # Snap to half-integers
                U_225_s = np.round(U_225 * 2) / 2
                V_225_s = np.round(V_225 * 2) / 2
                W_225_s = np.round(W_225 * 2) / 2
                
                # Verify the 2x2x5 solution
                def verify_225(U_mat, V_mat, W_mat):
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                s = np.sum(U_mat[i*m2+j] * V_mat[j*p2+k] * W_mat[k*n2+i])
                                if abs(s - 1.0) > 1e-9:
                                    return False
                    for a in range(nm2):
                        for b in range(mp2):
                            for c in range(pn2):
                                i_a = a // m2; j_a = a % m2
                                j_b = b // p2; k_b = b % p2
                                k_c = c // n2; i_c = c % n2
                                if not (j_a == j_b and k_b == k_c and i_a == i_c):
                                    continue
                                s = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                                if abs(s) > 1e-9:
                                    return False
                    return True
                
                if verify_225(U_225_s, V_225_s, W_225_s):
                    # Build the 2x4x5 solution with rank 34
                    total_rank_34 = 2 * target_rank_225  # 34
                    U34 = np.zeros((nm, total_rank_34), dtype=np.float64)
                    V34 = np.zeros((mp, total_rank_34), dtype=np.float64)
                    W34 = np.zeros((pn, total_rank_34), dtype=np.float64)
                    
                    # Block A: j in {0,1}
                    for i in range(n2):
                        for j in range(m2):
                            U34[i*m + j, :target_rank_225] = U_225_s[i*m2+j]
                        for j in range(m2):
                            for k in range(p2):
                                V34[j*p + k, :target_rank_225] = V_225_s[j*p2+k]
                        for k in range(p2):
                            for i in range(n2):
                                W34[k*n + i, :target_rank_225] = W_225_s[k*n2+i]
                    
                    # Block B: j in {2,3}
                    for i in range(n2):
                        for j in range(m2):
                            U34[i*m + (j+2), target_rank_225:] = U_225_s[i*m2+j]
                        for j in range(m2):
                            for k in range(p2):
                                V34[(j+2)*p + k, target_rank_225:] = V_225_s[j*p2+k]
                        for k in range(p2):
                            for i in range(n2):
                                W34[k*n + i, target_rank_225:] += W_225_s[k*n2+i]
                    
                    # Verify the full 2x4x5 solution
                    if verify_decomposition(U34, V34, W34):
                        U34_s = np.round(U34 * 2) / 2
                        V34_s = np.round(V34 * 2) / 2
                        W34_s = np.round(W34 * 2) / 2
                        
                        if verify_decomposition(U34_s, V34_s, W34_s):
                            U_list = [[int(round(U34_s[i, r] * 2)) / 2 for r in range(total_rank_34)] for i in range(nm)]
                            V_list = [[int(round(V34_s[i, r] * 2)) / 2 for r in range(total_rank_34)] for i in range(mp)]
                            W_list = [[int(round(W34_s[i, r] * 2)) / 2 for r in range(total_rank_34)] for i in range(pn)]
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = 34
                            return best_result
                            
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
