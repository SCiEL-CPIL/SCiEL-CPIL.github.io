import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 110.0

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

    # Strassen-based decomposition for 2x4x5
    # Rank 36
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

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
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)
    assert col == total_rank

    def verify_decomposition(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i * m + j]
                    v_val = V_mat[j * p + k]
                    w_val = W_mat[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    # Prepare rank 36 result
    U36 = U.copy()
    V36 = V.copy()
    W36 = W.copy()
    
    if verify_decomposition(U36, V36, W36):
        U_list = [[int(round(U36[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V36[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W36[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        best_result = {"U": U_list, "V": V_list, "W": W_list}
        best_rank = 36

    # Attempt Rank 35 Search
    if time.time() < deadline - 20.0:
        try:
            from scipy.optimize import least_squares
            
            # Define the tensor entries that must be 1 and those that must be 0
            # Indices: (i, j, k) where i in [0, n), j in [0, m), k in [0, p)
            # U index: i*m + j
            # V index: j*p + k
            # W index: k*n + i
            
            target_ones = []
            target_zeros = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target_ones.append((i*m+j, j*p+k, k*n+i))
            
            # We only need to check a subset of zeros to speed up, but for safety in verification we check all.
            # For the optimization, we can sample zeros or use all. 2*4*5=40 ones. Total entries 8*20*10=1600.
            # Zeros count: 1600 - 40 = 1560.
            # This is a large system. We will use a subset of zeros for the optimization to keep it fast,
            # and verify all at the end.
            
            zero_indices = []
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        i_a = a // m; j_a = a % m
                        j_b = b // p; k_b = b % p
                        k_c = c // n; i_c = c % n
                        if not (j_a == j_b and k_b == k_c and i_a == i_c):
                            zero_indices.append((a, b, c))
            
            # Shuffle zero indices and take a subset (e.g., 200) to reduce cost
            np.random.seed(42)
            subset_size = min(200, len(zero_indices))
            zero_subset_indices = np.random.choice(len(zero_indices), subset_size, replace=False)
            zero_subset = [zero_indices[i] for i in zero_subset_indices]
            
            def residuals(x):
                U_r, V_r, W_r = x[:nm*35], x[nm*35:nm*35+mp*35], x[nm*35+mp*35:nm*35+mp*35+pn*35]
                U_mat = U_r.reshape(nm, 35)
                V_mat = V_r.reshape(mp, 35)
                W_mat = W_r.reshape(pn, 35)
                
                res = []
                # Ones
                for a, b, c in target_ones:
                    val = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                    res.append(val - 1.0)
                # Zeros (subset)
                for a, b, c in zero_subset:
                    val = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                    res.append(val)
                return res

            # Initialize with Rank 36 solution, dropping one column
            # Try dropping each column? Or just the last one?
            # Let's try dropping the last column of the 36-rank solution.
            init_x = np.concatenate([U36[:, :-1].flatten(), V36[:, :-1].flatten(), W36[:, :-1].flatten()])
            
            best_res_norm = np.inf
            best_x = None
            
            # Try a few restarts
            for attempt in range(5):
                if time.time() > deadline - 5.0:
                    break
                
                # Perturb initialization slightly
                if attempt > 0:
                    init_x = init_x + np.random.normal(0, 0.1, size=len(init_x))
                
                try:
                    sol = least_squares(residuals, init_x, method='trf', max_nfev=500)
                    res_norm = np.linalg.norm(sol.fun)
                    if res_norm < best_res_norm:
                        best_res_norm = res_norm
                        best_x = sol.x
                        if res_norm < 1e-8:
                            break
                except Exception:
                    pass
            
            if best_x is not None and best_res_norm < 1e-7:
                U35 = best_x[:nm*35].reshape(nm, 35)
                V35 = best_x[nm*35:nm*35+mp*35].reshape(mp, 35)
                W35 = best_x[nm*35+mp*35:nm*35+mp*35+pn*35].reshape(pn, 35)
                
                # Verify full zero set
                if verify_decomposition(U35, V35, W35):
                    # Round to half-integers? The problem asks for integer/half-integer.
                    # If the solution is close to such values, we can snap.
                    # But if it's irrational, we can't. 
                    # The task says "integer/half-integer entries".
                    # If the optimal solution is not half-integer, we cannot use it.
                    # However, if rank 35 is possible, it's likely rational.
                    # Let's check if we can snap to half-integers.
                    U35_snapped = np.round(U35 * 2) / 2
                    V35_snapped = np.round(V35 * 2) / 2
                    W35_snapped = np.round(W35 * 2) / 2
                    
                    if verify_decomposition(U35_snapped, V35_snapped, W35_snapped):
                        U_list = [[int(round(U35_snapped[i, r] * 2)) / 2 for r in range(35)] for i in range(nm)]
                        V_list = [[int(round(V35_snapped[i, r] * 2)) / 2 for r in range(35)] for i in range(mp)]
                        W_list = [[int(round(W35_snapped[i, r] * 2)) / 2 for r in range(35)] for i in range(pn)]
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
        except Exception:
            pass

    return best_result
# EVOLVE-BLOCK-END
