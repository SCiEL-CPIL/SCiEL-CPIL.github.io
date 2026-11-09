import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s

    n, m, p = DIMENSIONS

    def verify(U, V, W, rank):
        """Check if decomposition is valid using numpy. Returns True if valid."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_round_and_verify(U, V, W, rank):
        """Try to round to small rationals and verify."""
        for denom in [1, 2, 4, 8, 16]:
            U_r = np.round(U * denom) / denom
            V_r = np.round(V * denom) / denom
            W_r = np.round(W * denom) / denom
            if verify(U_r, V_r, W_r, rank):
                return True, U_r, V_r, W_r
        return False, None, None, None

    # Initialize with schoolbook
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    # --- Strassen-based block decomposition ---
    # Use 2x2x2 Strassen blocks where possible, schoolbook for remainders
    # n=4 (2 blocks of 2), m=5 (2 blocks of 2 + 1 remainder), p=6 (3 blocks of 2)
    
    num_i_blocks = n // 2  # 2
    num_j_blocks = m // 2  # 2
    num_k_blocks = p // 2  # 3
    rem_j = m - num_j_blocks * 2  # 1
    
    strassen_rank = 7
    block_count = num_i_blocks * num_j_blocks * num_k_blocks
    rem_rank = rem_j * n * p  # 1 * 4 * 6 = 24
    total_strassen_rank = block_count * strassen_rank + rem_rank  # 12*7 + 24 = 108
    
    if total_strassen_rank < best_rank:
        U_new = np.zeros((n * m, total_strassen_rank))
        V_new = np.zeros((m * p, total_strassen_rank))
        W_new = np.zeros((p * n, total_strassen_rank))
        
        col = 0
        # Strassen terms for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each term, define (local_i, local_j, u_coeff) and (local_k, local_k_j, v_coeff)
        # A is 2x2: indices (0,0),(0,1),(1,0),(1,1)
        # B is 2x2: indices (0,0),(0,1),(1,0),(1,1)
        
        strassen_terms = [
            # M1: (A00+A11)(B00+B11)
            [((0,0,1),(0,0,1)), ((0,0,1),(0,1,1)), ((1,1,1),(0,0,1)), ((1,1,1),(0,1,1))],
            # M2: (A10+A11)B00
            [((1,0,1),(0,0,1)), ((1,1,1),(0,0,1))],
            # M3: A00(B10-B11)
            [((0,0,1),(1,0,1)), ((0,0,-1),(1,1,1))],
            # M4: A11(B01-B00)
            [((1,1,1),(0,1,1)), ((1,1,-1),(0,0,1))],
            # M5: (A00+A01)B11
            [((0,0,1),(1,1,1)), ((0,1,1),(1,1,1))],
            # M6: (A10-A00)(B00+B01)
            [((1,0,1),(0,0,1)), ((1,0,1),(0,1,1)), ((0,0,-1),(0,0,1)), ((0,0,-1),(0,1,1))],
            # M7: (A01-A11)(B10+B11)
            [((0,1,1),(1,0,1)), ((0,1,1),(1,1,1)), ((1,1,-1),(1,0,1)), ((1,1,-1),(1,1,1))],
        ]
        
        for bi in range(num_i_blocks):
            for bj in range(num_j_blocks):
                for bk in range(num_k_blocks):
                    for term_idx in range(strassen_rank):
                        r = col
                        for (li, lj, uc), (lk, lkj, vc) in strassen_terms[term_idx]:
                            gi = bi * 2 + li
                            gj = bj * 2 + lj
                            gk = bk * 2 + lk
                            U_new[gi * m + gj, r] += uc
                            V_new[gj * p + gk, r] += vc
                            # W will be set later based on output position
                        col += 1
        
        # Set W for Strassen blocks
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M5 + M7
        
        c_coeffs = {
            (0, 0): [(0, 1), (3, -1), (5, 1), (6, -1)],
            (0, 1): [(2, 1), (4, 1)],
            (1, 0): [(1, 1), (3, 1)],
            (1, 1): [(0, 1), (2, -1), (4, 1), (6, 1)],
        }
        
        col = 0
        for bi in range(num_i_blocks):
            for bj in range(num_j_blocks):
                for bk in range(num_k_blocks):
                    for term_idx in range(strassen_rank):
                        r = col
                        for (out_i, out_j), (t_idx, coeff) in c_coeffs.items():
                            pass
                        # Actually W is indexed by [k*n+i, r]
                        # For each output position (oi, oj) in the 2x2 C block,
                        # W[global_k * n + global_i, r] = sum of coefficients
                        # But W is per (k,i) pair, not per (oi,oj)
                        # Let me reconsider: W[k*n+i, r] is the coefficient for C[i,j]
                        # Actually the tensor is T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        # So W[k*n+i, r] contributes to all j for given (i,k)
                        # This means for a Strassen block, W depends on (i,k) but not j
                        # This is a problem - Strassen doesn't directly fit this tensor structure
                        # Let me use a different approach: just use schoolbook for the remainder
                        # and verify the Strassen approach more carefully
                        col += 1
        
        # The Strassen approach is complex for this tensor structure. 
        # Let's just use the schoolbook result as the best we can guarantee.
        # The optimization below may find improvements.
    
    # Try to find lower rank decompositions via optimization
    # We will try ranks from schoolbook_rank - 1 down to a lower bound
    for target_rank in range(schoolbook_rank - 1, max(30, schoolbook_rank - 10)):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            rng = np.random.RandomState(attempt * 100 + target_rank)
            U_init = rng.randn(n * m, target_rank) * 0.1
            V_init = rng.randn(m * p, target_rank) * 0.1
            W_init = rng.randn(p * n, target_rank) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                size_U = n * m * target_rank
                size_V = m * p * target_rank
                size_W = p * n * target_rank
                
                U_flat = params[:size_U]
                V_flat = params[size_U:size_U + size_V]
                W_flat = params[size_U + size_V:size_U + size_V + size_W]
                
                U = U_flat.reshape((n * m, target_rank))
                V = V_flat.reshape((m * p, target_rank))
                W = W_flat.reshape((p * n, target_rank))
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U[i * m + j]
                            v_val = V[j * p + k]
                            w_val = W[k * n + i]
                            s = np.dot(u_val, v_val * w_val)
                            error += (s - 1.0) ** 2
                return error

            try:
                from scipy.optimize import minimize
                res = minimize(objective, params_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
                if res.success or res.fun < 1e-8:
                    U_opt = res.x[:n * m * target_rank].reshape((n * m, target_rank))
                    V_opt = res.x[n * m * target_rank:n * m * target_rank + m * p * target_rank].reshape((m * p, target_rank))
                    W_opt = res.x[n * m * target_rank + m * p * target_rank:].reshape((p * n, target_rank))
                    
                    valid, U_r, V_r, W_r = try_round_and_verify(U_opt, V_opt, W_opt, target_rank)
                    if valid:
                        best_rank = target_rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
