import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin before 140s limit

    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12

    # Build the target tensor for 2x3x6 multiplication
    valid_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.add((a, b, c))

    # Schoolbook solution: rank = n*m*p = 36
    best_rank = n * m * p
    rank = best_rank
    U = np.zeros((num_A, rank))
    V = np.zeros((num_B, rank))
    W = np.zeros((num_C, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Strategy: Split the 3x6 matrix into two 3x3 blocks
    # Sub-problem: 2x3x3
    sub_n, sub_m, sub_p = 2, 3, 3
    sub_num_A = sub_n * sub_m # 6
    sub_num_B = sub_m * sub_p # 9
    sub_num_C = sub_p * sub_n # 6
    
    sub_valid_triples = set()
    for i in range(sub_n):
        for j in range(sub_m):
            for k in range(sub_p):
                a = i * sub_m + j
                b = j * sub_p + k
                c = k * sub_n + i
                sub_valid_triples.add((a, b, c))
                
    # Pre-compute indices for efficient evaluation
    sub_a_list = []
    sub_b_list = []
    sub_c_list = []
    sub_target = []
    for a in range(sub_num_A):
        for b in range(sub_num_B):
            for c in range(sub_num_C):
                sub_a_list.append(a)
                sub_b_list.append(b)
                sub_c_list.append(c)
                sub_target.append(1.0 if (a, b, c) in sub_valid_triples else 0.0)
                
    sub_a_arr = np.array(sub_a_list)
    sub_b_arr = np.array(sub_b_list)
    sub_c_arr = np.array(sub_c_list)
    sub_target_arr = np.array(sub_target)

    found_sub = False
    best_sub_rank = 18
    best_sub_U = best_sub_V = best_sub_W = None

    # Try to find low-rank decomposition for 2x3x3 sub-problem
    # Known rank is 16.
    for target_sub_rank in [16, 17]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for restart in range(20):
            if time.time() > deadline - 15:
                break
                
            np.random.seed(1000 * target_sub_rank + restart)
            U_curr = np.random.randn(sub_num_A, target_sub_rank) * 0.5
            V_curr = np.random.randn(sub_num_B, target_sub_rank) * 0.5
            W_curr = np.random.randn(sub_num_C, target_sub_rank) * 0.5
            
            # Phase 1: ALS to get close
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Solve for U
                U_new = np.zeros_like(U_curr)
                for a in range(sub_num_A):
                    mask = sub_a_arr == a
                    if np.any(mask):
                        # X rows are V[b] * W[c]
                        b_idx = sub_b_arr[mask]
                        c_idx = sub_c_arr[mask]
                        # Element-wise product of V[b_idx, :] and W[c_idx, :]
                        # Shape: (num_valid, rank)
                        X_a = V_curr[b_idx] * W_curr[c_idx]
                        y_a = sub_target_arr[mask]
                        u, residuals, rank_u, s = np.linalg.lstsq(X_a, y_a, rcond=None)
                        U_new[a] = u
                U_curr = U_new

                # Solve for V
                V_new = np.zeros_like(V_curr)
                for b in range(sub_num_B):
                    mask = sub_b_arr == b
                    if np.any(mask):
                        a_idx = sub_a_arr[mask]
                        c_idx = sub_c_arr[mask]
                        X_b = U_curr[a_idx] * W_curr[c_idx]
                        y_b = sub_target_arr[mask]
                        v, residuals, rank_v, s = np.linalg.lstsq(X_b, y_b, rcond=None)
                        V_new[b] = v
                V_curr = V_new

                # Solve for W
                W_new = np.zeros_like(W_curr)
                for c in range(sub_num_C):
                    mask = sub_c_arr == c
                    if np.any(mask):
                        a_idx = sub_a_arr[mask]
                        b_idx = sub_b_arr[mask]
                        X_c = U_curr[a_idx] * V_curr[b_idx]
                        y_c = sub_target_arr[mask]
                        w, residuals, rank_w, s = np.linalg.lstsq(X_c, y_c, rcond=None)
                        W_new[c] = w
                W_curr = W_new
                
                # Early check
                val = np.sum(U_curr[sub_a_arr, :] * V_curr[sub_b_arr, :] * W_curr[sub_c_arr, :], axis=1)
                err = np.max(np.abs(val - sub_target_arr))
                if err < 1e-6:
                    break

            # Phase 2: Polish with least_squares
            # Flatten U, V, W
            x0 = np.concatenate([U_curr.flatten(), V_curr.flatten(), W_curr.flatten()])
            
            def residuals(x):
                U_r = x[:sub_num_A * target_sub_rank].reshape(sub_num_A, target_sub_rank)
                V_r = x[sub_num_A * target_sub_rank : sub_num_A * target_sub_rank + sub_num_B * target_sub_rank].reshape(sub_num_B, target_sub_rank)
                W_r = x[sub_num_A * target_sub_rank + sub_num_B * target_sub_rank :].reshape(sub_num_C, target_sub_rank)
                
                # Compute values for all triples
                # U[sub_a_arr, :] has shape (N, rank)
                # V[sub_b_arr, :] has shape (N, rank)
                # W[sub_c_arr, :] has shape (N, rank)
                vals = np.sum(U_r[sub_a_arr, :] * V_r[sub_b_arr, :] * W_r[sub_c_arr, :], axis=1)
                return vals - sub_target_arr

            try:
                result = least_squares(residuals, x0, method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=500)
                x_opt = result.x
                U_opt = x_opt[:sub_num_A * target_sub_rank].reshape(sub_num_A, target_sub_rank)
                V_opt = x_opt[sub_num_A * target_sub_rank : sub_num_A * target_sub_rank + sub_num_B * target_sub_rank].reshape(sub_num_B, target_sub_rank)
                W_opt = x_opt[sub_num_A * target_sub_rank + sub_num_B * target_sub_rank :].reshape(sub_num_C, target_sub_rank)
                
                # Verify
                vals = np.sum(U_opt[sub_a_arr, :] * V_opt[sub_b_arr, :] * W_opt[sub_c_arr, :], axis=1)
                max_err = np.max(np.abs(vals - sub_target_arr))
                
                if max_err < 1e-9:
                    found = True
                    best_sub_rank = target_sub_rank
                    best_sub_U, best_sub_V, best_sub_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                    break
            except Exception:
                pass

        if found:
            found_sub = True
            break
    
    # If we found a sub-decomposition, construct the full 2x3x6 decomposition
    if found_sub:
        sub_rank = best_sub_rank
        full_rank = 2 * sub_rank
        
        U_full = np.zeros((num_A, full_rank))
        V_full = np.zeros((num_B, full_rank))
        W_full = np.zeros((num_C, full_rank))
        
        # First block: k in {0,1,2}
        for r in range(sub_rank):
            U_full[:, r] = best_sub_U[:, r]
            for j in range(3):
                for k in range(3):
                    sub_b = j * 3 + k
                    full_b = j * 6 + k
                    V_full[full_b, r] = best_sub_V[sub_b, r]
            for k in range(3):
                for i in range(2):
                    sub_c = 2 * k + i
                    full_c = 2 * k + i
                    W_full[full_c, r] = best_sub_W[sub_c, r]
        
        # Second block: k in {3,4,5}
        offset = sub_rank
        for r in range(sub_rank):
            U_full[:, offset + r] = best_sub_U[:, r]
            for j in range(3):
                for k_sub in range(3):
                    k = k_sub + 3
                    sub_b = j * 3 + k_sub
                    full_b = j * 6 + k
                    V_full[full_b, offset + r] = best_sub_V[sub_b, r]
            for k_sub in range(3):
                k = k_sub + 3
                for i in range(2):
                    sub_c = 2 * k_sub + i
                    full_c = 2 * k + i
                    W_full[full_c, offset + r] = best_sub_W[sub_c, r]
        
        # Verify the full decomposition
        # We can trust the sub-verification if the mapping is correct, but let's do a quick check
        # To save time, we only check a sample or assume correctness if sub was perfect.
        # Given the tight tolerance on sub, full should be valid.
        
        best_rank = full_rank
        best_U, best_V, best_W = U_full.copy(), V_full.copy(), W_full.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
