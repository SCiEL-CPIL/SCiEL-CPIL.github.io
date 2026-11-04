import numpy as np
import scipy.optimize as so
import time
import sys

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Build target tensor T indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                indices.append((a, b, w))
    
    idx_a = np.array([x[0] for x in indices], dtype=np.int32)
    idx_b = np.array([x[1] for x in indices], dtype=np.int32)
    idx_w = np.array([x[2] for x in indices], dtype=np.int32)
    
    # Precompute full tensor and zero mask
    full_T = np.zeros((dim_A, dim_B, dim_C))
    full_T[idx_a, idx_b, idx_w] = 1.0
    zero_mask = (full_T == 0)
    zero_flat = full_T[zero_mask]  # all zeros
    num_zeros = np.sum(zero_mask)
    
    def residual_full(theta, r):
        """Residual for the full tensor decomposition."""
        U = theta[:dim_A * r].reshape(dim_A, r)
        V = theta[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
        W = theta[dim_A * r + dim_B * r :].reshape(dim_C, r)
        
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        vals = hat_T[idx_a, idx_b, idx_w]
        res_sparse = vals - 1.0
        zero_vals = hat_T[zero_mask]
        return np.concatenate([res_sparse, 0.1 * zero_vals])
    
    def residual_residual(theta, r, term_a, term_b, term_w):
        """
        Residual for decomposing T - term, where term = outer(term_a, term_b, term_w).
        We want: sum_r u_r v_r w_r = T - term
        So the residual is: hat_T - (T - term) = hat_T - T + term
        On sparse indices: hat_T - 1 + term_val
        On zero indices: hat_T + term_val
        """
        U = theta[:dim_A * r].reshape(dim_A, r)
        V = theta[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
        W = theta[dim_A * r + dim_B * r :].reshape(dim_C, r)
        
        hat_T = np.einsum('ar,br,wr->abw', U, V, W)
        
        # Term values
        term_tensor = np.einsum('a,b,w->abw', term_a, term_b, term_w)
        
        # On sparse indices: residual = hat_T - 1 + term
        vals = hat_T[idx_a, idx_b, idx_w]
        term_vals = term_tensor[idx_a, idx_b, idx_w]
        res_sparse = vals - 1.0 + term_vals
        
        # On zero indices: residual = hat_T + term (since T=0 there)
        zero_vals = hat_T[zero_mask]
        term_zeros = term_tensor[zero_mask]
        res_zero = zero_vals + term_zeros
        
        return np.concatenate([res_sparse, 0.1 * res_zero])
    
    # Fallback solution (Rank 42) using Strassen
    def get_fallback():
        U = [[0] * 42 for _ in range(n * m)]
        V = [[0] * 42 for _ in range(m * p)]
        W = [[0] * 42 for _ in range(p * n)]
        
        def add_strassen_block(a_idx, b_idx, w_idx, start_col):
            U[a_idx[0]][start_col + 0] += 1
            U[a_idx[3]][start_col + 0] += 1
            U[a_idx[2]][start_col + 1] += 1
            U[a_idx[3]][start_col + 1] += 1
            U[a_idx[0]][start_col + 2] += 1
            U[a_idx[3]][start_col + 3] += 1
            U[a_idx[0]][start_col + 4] += 1
            U[a_idx[1]][start_col + 4] += 1
            U[a_idx[2]][start_col + 5] += 1
            U[a_idx[0]][start_col + 5] -= 1
            U[a_idx[1]][start_col + 6] += 1
            U[a_idx[3]][start_col + 6] -= 1
            
            V[b_idx[0]][start_col + 0] += 1
            V[b_idx[3]][start_col + 0] += 1
            V[b_idx[0]][start_col + 1] += 1
            V[b_idx[1]][start_col + 2] += 1
            V[b_idx[3]][start_col + 2] -= 1
            V[b_idx[2]][start_col + 3] += 1
            V[b_idx[0]][start_col + 3] -= 1
            V[b_idx[3]][start_col + 4] += 1
            V[b_idx[0]][start_col + 5] += 1
            V[b_idx[1]][start_col + 5] += 1
            V[b_idx[2]][start_col + 6] += 1
            V[b_idx[3]][start_col + 6] += 1
            
            W[w_idx[0]][start_col + 0] += 1
            W[w_idx[0]][start_col + 3] += 1
            W[w_idx[0]][start_col + 4] -= 1
            W[w_idx[0]][start_col + 6] += 1
            
            W[w_idx[1]][start_col + 2] += 1
            W[w_idx[1]][start_col + 4] += 1
            
            W[w_idx[2]][start_col + 1] += 1
            W[w_idx[2]][start_col + 3] += 1
            
            W[w_idx[3]][start_col + 0] += 1
            W[w_idx[3]][start_col + 1] -= 1
            W[w_idx[3]][start_col + 2] += 1
            W[w_idx[3]][start_col + 5] += 1

        a_L = [0, 1, 4, 5]
        a_R = [2, 3, 6, 7]
        
        b_1_T = [0, 1, 6, 7]
        b_1_B = [12, 13, 18, 19]
        w_1 = [0, 2, 1, 3]
        add_strassen_block(a_L, b_1_T, w_1, 0)
        add_strassen_block(a_R, b_1_B, w_1, 7)
        
        b_2_T = [2, 3, 8, 9]
        b_2_B = [14, 15, 20, 21]
        w_2 = [4, 6, 5, 7]
        add_strassen_block(a_L, b_2_T, w_2, 14)
        add_strassen_block(a_R, b_2_B, w_2, 21)
        
        b_3_T = [4, 5, 10, 11]
        b_3_B = [16, 17, 22, 23]
        w_3 = [8, 10, 9, 11]
        add_strassen_block(a_L, b_3_T, w_3, 28)
        add_strassen_block(a_R, b_3_B, w_3, 35)
        
        return U, V, W

    U_fb, V_fb, W_fb = get_fallback()
    
    start_time = time.time()
    deadline = 110.0
    
    def verify_solution(U_mat, V_mat, W_mat):
        """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
        hat_T = np.einsum('ar,br,wr->abw', U_mat, V_mat, W_mat)
        vals = hat_T[idx_a, idx_b, idx_w]
        if not np.allclose(vals, 1.0, atol=1e-9):
            return False
        zero_vals = hat_T[zero_mask]
        if not np.allclose(zero_vals, 0.0, atol=1e-9):
            return False
        return True
    
    def try_round_and_verify(U_sol, V_sol, W_sol, r):
        """Round to half-integers and verify. Return (U,V,W) or None."""
        U_int = np.round(U_sol * 2) / 2
        V_int = np.round(V_sol * 2) / 2
        W_int = np.round(W_sol * 2) / 2
        
        if verify_solution(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        # Local search: flip entries by +/-0.5
        best_cost = np.inf
        best_U, best_V, best_W = U_int.copy(), V_int.copy(), W_int.copy()
        
        for _ in range(100):
            if time.time() > deadline:
                break
            
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                row, col = np.random.randint(dim_A), np.random.randint(r)
                delta = 0.5 if np.random.random() < 0.5 else -0.5
                U_int[row, col] += delta
            elif mat_idx == 1:
                row, col = np.random.randint(dim_B), np.random.randint(r)
                delta = 0.5 if np.random.random() < 0.5 else -0.5
                V_int[row, col] += delta
            else:
                row, col = np.random.randint(dim_C), np.random.randint(r)
                delta = 0.5 if np.random.random() < 0.5 else -0.5
                W_int[row, col] += delta
            
            hat_T_test = np.einsum('ar,br,wr->abw', U_int, V_int, W_int)
            vals_test = hat_T_test[idx_a, idx_b, idx_w]
            zero_test = hat_T_test[zero_mask]
            test_cost = np.sum((vals_test - 1.0)**2) + 0.01 * np.sum(zero_test**2)
            
            if test_cost < best_cost:
                best_cost = test_cost
                best_U, best_V, best_W = U_int.copy(), V_int.copy(), W_int.copy()
            else:
                if mat_idx == 0:
                    U_int[row, col] -= delta
                elif mat_idx == 1:
                    V_int[row, col] -= delta
                else:
                    W_int[row, col] -= delta
        
        U_int, V_int, W_int = best_U, best_V, best_W
        
        if verify_solution(U_int, V_int, W_int):
            return U_int, V_int, W_int
        return None
    
    # Strategy 1: Direct search on full tensor (original approach)
    for r in range(24, 42):
        if time.time() > deadline:
            break
        
        for attempt in range(15):
            if time.time() > deadline:
                break
            
            np.random.seed(r * 1000 + attempt)
            scale = 1.0 + 0.3 * (attempt % 5)
            U0 = np.random.randn(dim_A, r) * scale
            V0 = np.random.randn(dim_B, r) * scale
            W0 = np.random.randn(dim_C, r) * scale
            theta0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = so.least_squares(lambda t: residual_full(t, r), theta0, method='lm', 
                                       xtol=1e-10, ftol=1e-10, gtol=1e-10)
                
                if res.success and res.cost < 1e-6:
                    U_sol = res.x[:dim_A * r].reshape(dim_A, r)
                    V_sol = res.x[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                    W_sol = res.x[dim_A * r + dim_B * r :].reshape(dim_C, r)
                    
                    result = try_round_and_verify(U_sol, V_sol, W_sol, r)
                    if result is not None:
                        return {"U": result[0].tolist(), "V": result[1].tolist(), "W": result[2].tolist()}
            except Exception:
                pass
    
    # Strategy 2: Sub-tensor splitting - extract a rank-1 term, solve residual
    # Pick a specific (i,j,k) triple to extract as a rank-1 term
    # The term for (i,j,k) has:
    #   a = i*m + j, b = j*p + k, w = k*n + i
    # We try different choices of the extracted term
    
    candidate_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                w = k * n + i
                term_a = np.zeros(dim_A)
                term_b = np.zeros(dim_B)
                term_w = np.zeros(dim_C)
                term_a[a] = 1.0
                term_b[b] = 1.0
                term_w[w] = 1.0
                candidate_terms.append((term_a, term_b, term_w, (i, j, k)))
    
    # Try sub-tensor approach for various ranks
    for r in range(20, 42):
        if time.time() > deadline:
            break
        
        # Try a subset of candidate terms (not all, to save time)
        terms_to_try = candidate_terms[:min(6, len(candidate_terms))]
        
        for term_a, term_b, term_w, triple in terms_to_try:
            if time.time() > deadline:
                break
            
            for attempt in range(8):
                if time.time() > deadline:
                    break
                
                np.random.seed(10000 + r * 100 + attempt * 7 + hash(triple) % 1000)
                scale = 1.0 + 0.4 * (attempt % 3)
                U0 = np.random.randn(dim_A, r) * scale
                V0 = np.random.randn(dim_B, r) * scale
                W0 = np.random.randn(dim_C, r) * scale
                theta0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
                
                try:
                    res = so.least_squares(
                        lambda t: residual_residual(t, r, term_a, term_b, term_w),
                        theta0, method='lm', xtol=1e-10, ftol=1e-10, gtol=1e-10
                    )
                    
                    if res.success and res.cost < 1e-6:
                        U_sol = res.x[:dim_A * r].reshape(dim_A, r)
                        V_sol = res.x[dim_A * r : dim_A * r + dim_B * r].reshape(dim_B, r)
                        W_sol = res.x[dim_A * r + dim_B * r :].reshape(dim_C, r)
                        
                        # The solution gives: sum_r u_r v_r w_r = T - term
                        # So the full decomposition is: term + sum_r u_r v_r w_r
                        # Add the term as an extra rank-1 component
                        U_full = np.column_stack([U_sol, term_a])
                        V_full = np.column_stack([V_sol, term_b])
                        W_full = np.column_stack([W_sol, term_w])
                        
                        # Now we have rank r+1 solution. Try to round.
                        U_int = np.round(U_full * 2) / 2
                        V_int = np.round(V_full * 2) / 2
                        W_int = np.round(W_full * 2) / 2
                        
                        if verify_solution(U_int, V_int, W_int):
                            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                        
                        # Local polish
                        r_full = r + 1
                        best_cost = np.inf
                        best_U, best_V, best_W = U_int.copy(), V_int.copy(), W_int.copy()
                        
                        for _ in range(80):
                            if time.time() > deadline:
                                break
                            
                            mat_idx = np.random.randint(3)
                            if mat_idx == 0:
                                row, col = np.random.randint(dim_A), np.random.randint(r_full)
                                delta = 0.5 if np.random.random() < 0.5 else -0.5
                                U_int[row, col] += delta
                            elif mat_idx == 1:
                                row, col = np.random.randint(dim_B), np.random.randint(r_full)
                                delta = 0.5 if np.random.random() < 0.5 else -0.5
                                V_int[row, col] += delta
                            else:
                                row, col = np.random.randint(dim_C), np.random.randint(r_full)
                                delta = 0.5 if np.random.random() < 0.5 else -0.5
                                W_int[row, col] += delta
                            
                            hat_T_test = np.einsum('ar,br,wr->abw', U_int, V_int, W_int)
                            vals_test = hat_T_test[idx_a, idx_b, idx_w]
                            zero_test = hat_T_test[zero_mask]
                            test_cost = np.sum((vals_test - 1.0)**2) + 0.01 * np.sum(zero_test**2)
                            
                            if test_cost < best_cost:
                                best_cost = test_cost
                                best_U, best_V, best_W = U_int.copy(), V_int.copy(), W_int.copy()
                            else:
                                if mat_idx == 0:
                                    U_int[row, col] -= delta
                                elif mat_idx == 1:
                                    V_int[row, col] -= delta
                                else:
                                    W_int[row, col] -= delta
                        
                        U_int, V_int, W_int = best_U, best_V, best_W
                        
                        if verify_solution(U_int, V_int, W_int):
                            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                except Exception:
                    pass
    
    # If no better rank found, return the known rank 42 solution
    return {"U": U_fb, "V": V_fb, "W": W_fb}
